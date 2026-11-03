import ast, json, time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import streamlit as st

MODEL = Path(__file__).parent / "models" / "yolo11n.onnx"
SIZE = 640
COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

st.set_page_config(page_title="EdgeAI · Cloud Demo", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp{background:#0b1018}.block-container{max-width:1280px;padding-top:2rem}
[data-testid="stSidebar"]{background:#101823}
[data-testid="stMetric"]{background:#111a27;border:1px solid #26364b;border-radius:12px;padding:14px}
.kicker{color:#77a9ff;font-size:.78rem;font-weight:700;letter-spacing:.15em}
.note{background:#111a27;border:1px solid #26364b;border-radius:12px;padding:14px}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    opts=ort.SessionOptions()
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads=2
    sess=ort.InferenceSession(path,sess_options=opts,providers=["CPUExecutionProvider"])
    inp=sess.get_inputs()[0]
    if inp.shape != [1,3,SIZE,SIZE]:
        raise RuntimeError(f"Expected [1,3,{SIZE},{SIZE}], got {inp.shape}")
    raw=sess.get_modelmeta().custom_metadata_map.get("names")
    try:
        parsed=json.loads(raw) if raw else None
    except json.JSONDecodeError:
        try: parsed=ast.literal_eval(raw)
        except Exception: parsed=None
    names={int(k):str(v) for k,v in parsed.items()} if isinstance(parsed,dict) else ({i:str(v) for i,v in enumerate(parsed)} if isinstance(parsed,list) else dict(enumerate(COCO)))
    return sess,inp.name,[o.name for o in sess.get_outputs()],names

def preprocess(im):
    h,w=im.shape[:2]; scale=min(SIZE/h,SIZE/w)
    rw,rh=round(w*scale),round(h*scale)
    resized=cv2.resize(im,(rw,rh),interpolation=cv2.INTER_LINEAR)
    pw,ph=SIZE-rw,SIZE-rh
    left,top=round(pw/2-.1),round(ph/2-.1)
    padded=cv2.copyMakeBorder(resized,top,ph-top,left,pw-left,cv2.BORDER_CONSTANT,value=(114,114,114))
    x=padded[:,:,::-1].transpose(2,0,1)[None]
    return np.ascontiguousarray(x,dtype=np.float32)/255.0,scale,left,top

def box_iou(one,many):
    lt=np.maximum(one[:2],many[:,:2]); rb=np.minimum(one[2:],many[:,2:])
    inter=np.prod(np.maximum(rb-lt,0),axis=1)
    a=np.prod(np.maximum(one[2:]-one[:2],0)); b=np.prod(np.maximum(many[:,2:]-many[:,:2],0),axis=1)
    return inter/np.maximum(a+b-inter,1e-9)

def decode(out,scale,left,top,w,h,names,conf,threshold):
    p=np.asarray(out,dtype=np.float32)
    if p.ndim==3: p=p[0]
    if p.shape[0]==4+len(names): p=p.T
    elif p.shape[0]<p.shape[1]: p=p.T
    scores0=p[:,4:]; ids=np.argmax(scores0,axis=1); scores=scores0[np.arange(len(ids)),ids]
    keep=np.flatnonzero(scores>=conf)
    if not len(keep): return []
    p,scores,ids=p[keep],scores[keep],ids[keep]
    c,sz=p[:,:2],p[:,2:4]; boxes=np.column_stack((c-sz/2,c+sz/2))
    selected=[]
    for cid in np.unique(ids):
        order=np.flatnonzero(ids==cid); order=order[np.argsort(scores[order])[::-1]]
        while len(order):
            i=int(order[0]); selected.append(i); rest=order[1:]
            if not len(rest): break
            order=rest[box_iou(boxes[i],boxes[rest])<=threshold]
    result=[]
    for i in sorted(selected,key=lambda j:float(scores[j]),reverse=True)[:100]:
        x1,y1,x2,y2=boxes[i]; cid=int(ids[i])
        result.append({"class":names.get(cid,str(cid)),"confidence":float(scores[i]),
          "box":(float(np.clip((x1-left)/scale,0,w)),float(np.clip((y1-top)/scale,0,h)),
                 float(np.clip((x2-left)/scale,0,w)),float(np.clip((y2-top)/scale,0,h)))})
    return result

def annotate(im,dets):
    out=im.copy()
    for d in dets:
        x1,y1,x2,y2=map(lambda x:int(round(x)),d["box"])
        color=(214,130,54); cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        label=f'{d["class"]} {d["confidence"]:.2f}'
        (_,th),base=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        y=max(th+base+6,y1)
        cv2.rectangle(out,(x1,max(0,y-th-base-6)),(x1+min(320,len(label)*11+8),y),color,-1)
        cv2.putText(out,label,(x1+4,y-base-3),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
    return out

st.markdown('<div class="kicker">EDGEAI · PUBLIC CLOUD DEMO</div>',unsafe_allow_html=True)
st.title("Vision inference, running on CPU.")
st.caption("YOLO11n · ONNX Runtime · Image detection")
st.markdown('<div class="note">Hosted inference runs on <b>CPU</b>. It does not access your laptop GPU, and its timings are not comparable to local CUDA benchmarks.</div>',unsafe_allow_html=True)
if not MODEL.is_file():
    st.error("Missing models/yolo11n.onnx"); st.stop()
with st.sidebar:
    st.markdown("### ◈ EdgeAI")
    confidence=st.slider("Confidence threshold",.05,.95,.25,.05)
    iou_threshold=st.slider("NMS IoU threshold",.1,.9,.7,.05)
    st.caption("YOLO11n FP32 · 640×640 · ONNX Runtime CPU")
upload=st.file_uploader("Upload an image",type=["jpg","jpeg","png"])
if upload:
    if upload.size>10*1024*1024: st.error("Image must be under 10 MB."); st.stop()
    image=cv2.imdecode(np.frombuffer(upload.getvalue(),np.uint8),cv2.IMREAD_COLOR)
    if image is None: st.error("Could not decode image."); st.stop()
    st.image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),caption="Input image",width="stretch")
    if st.button("Run detection",type="primary",width="stretch"):
        try:
            with st.spinner("Running ONNX inference on CPU…"):
                session,input_name,outputs,names=load_model(str(MODEL))
                tensor,scale,left,top=preprocess(image)
                start=time.perf_counter()
                pred=session.run(outputs,{input_name:tensor})
                runtime=(time.perf_counter()-start)*1000
                detections=decode(pred[0],scale,left,top,image.shape[1],image.shape[0],names,confidence,iou_threshold)
                result=annotate(image,detections)
            st.success("Inference completed")
            a,b,c=st.columns(3); a.metric("Objects detected",len(detections)); b.metric("ONNX Runtime",f"{runtime:.1f} ms"); c.metric("Provider","CPU")
            left_col,right_col=st.columns([1.3,1])
            with left_col:
                st.image(cv2.cvtColor(result,cv2.COLOR_BGR2RGB),caption="Annotated result",width="stretch")
                ok,encoded=cv2.imencode(".jpg",result)
                if ok: st.download_button("Download annotated image",encoded.tobytes(),"edgeai_annotated.jpg","image/jpeg")
            with right_col:
                st.subheader("Detection details")
                if detections:
                    st.dataframe([{"Class":d["class"],"Confidence":round(d["confidence"],3),"Box":tuple(round(v,1) for v in d["box"])} for d in detections],hide_index=True,width="stretch")
                else: st.info("No objects detected above this threshold.")
        except Exception as exc: st.exception(exc)
st.divider()
st.caption("EdgeAI · Cloud CPU demo. See the repository README for local CUDA inference and benchmarks.")
