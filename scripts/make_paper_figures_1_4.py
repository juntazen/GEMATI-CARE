import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.family"]="DejaVu Sans"

# ---------- Figure 1: architecture ----------
fig, ax = plt.subplots(figsize=(3.4, 4.6), dpi=300)
ax.set_xlim(0, 10); ax.set_ylim(0, 14); ax.axis("off")
def box(x,y,w,h,title,sub,ec,fc="white",ls="-",fs=6.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.25",ec=ec,fc=fc,lw=1.1,ls=ls))
    ax.text(x+w/2,y+h*0.66,title,ha="center",va="center",fontsize=fs,weight="bold",color="#1b2a3a")
    ax.text(x+w/2,y+h*0.30,sub,ha="center",va="center",fontsize=fs-1.2,color="#1b2a3a")
def arrow(x1,y1,x2,y2,ls="-"):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=0.9,color="#3b4a5a",ls=ls,mutation_scale=7))
X,W,H=0.3,5.6,1.25
steps=[("1  Input message","CRADLE text (low / medium / high)","#2a7ab0"),
       ("2  TF-IDF features","word 1–2-grams + char 3–5-grams","#5a5ad0"),
       ("3  Risk head","class-weighted OvR logistic regression","#1f9e89"),
       ("4  Temperature scaling","probability calibration, n = 418","#e39b12"),
       ("5  Mondrian set C(x)","class-wise quantiles, n = 836","#2a9fbf"),
       ("6  Minimax router",r"$a^*=\arg\min_a \max_{k\in C(x)} L(a,k)$","#d0457a")]
ys=[12.4-i*1.85 for i in range(len(steps))]
for (t,s,c),y in zip(steps,ys): box(X,y,W,H,t,s,c)
for y1,y2 in zip(ys[:-1],ys[1:]): arrow(X+W/2,y1,X+W/2,y2+H)
acts=[("SUPPORT","routine reply","#1f9e89"),("CLARIFY","ask / review","#e39b12"),("ESCALATE","human pathway","#d0457a")]
ay=0.15; aw=3.0
for i,(t,s,c) in enumerate(acts):
    ax.add_patch(FancyBboxPatch((0.15+i*3.27,ay),aw,1.05,boxstyle="round,pad=0.02,rounding_size=0.2",ec=c,fc="white",lw=1.2))
    ax.text(0.15+i*3.27+aw/2,ay+0.70,t,ha="center",va="center",fontsize=5.8,weight="bold",color="#1b2a3a")
    ax.text(0.15+i*3.27+aw/2,ay+0.33,s,ha="center",va="center",fontsize=4.8,color="#1b2a3a")
    arrow(X+W/2,ys[-1],0.15+i*3.27+aw/2,ay+1.05)
# audit panel
px,py,pw,ph=6.25,3.1,3.6,10.55
ax.add_patch(FancyBboxPatch((px,py),pw,ph,boxstyle="round,pad=0.02,rounding_size=0.25",ec="#7a8a9a",fc="#f1f5f9",lw=0.9,ls="--"))
ax.text(px+pw/2,py+ph-0.45,"Evaluation-only\naudits",ha="center",va="top",fontsize=5.8,weight="bold",color="#1b2a3a")
items=["5 repeated splits","1,000 bootstrap","cost / tie / α\nsensitivity","exact + semantic\nleakage audit","IndoSafety\n(5 language\nvariants)","SIM-VAIL\nmulti-turn","CPU FP32 /\nINT8 benchmark"]
for i,t in enumerate(items):
    ax.text(px+pw/2,py+ph-1.75-i*1.25,t,ha="center",va="center",fontsize=4.8,color="#1b2a3a")
ax.text(px+pw/2,py+0.35,"not part of the\ncoverage claim",ha="center",va="center",fontsize=4.3,style="italic",color="#8a2b2b")
fig.savefig("fig1_architecture.png",dpi=300,bbox_inches="tight",pad_inches=0.03)

# ---------- Figure 4: misses vs over-escalations ----------
pol=["Argmax","Expected cost","Minimax α=0.10","Minimax α=0.05"]
miss=[55,21,24,12]; over=[85,205,161,236]
fig,ax=plt.subplots(figsize=(3.4,2.3),dpi=300)
import numpy as np
y=np.arange(len(pol))
ax.barh(y+0.19,miss,height=0.36,color="#d0457a",label="Missed high-risk messages (of 234)")
ax.barh(y-0.19,over,height=0.36,color="#2a9fbf",label="Over-escalated messages (of 600)")
for i,(m,o) in enumerate(zip(miss,over)):
    ax.text(m+3,i+0.19,str(m),va="center",fontsize=5.5)
    ax.text(o+3,i-0.19,str(o),va="center",fontsize=5.5)
ax.set_yticks(y); ax.set_yticklabels(pol,fontsize=6); ax.invert_yaxis()
ax.set_xlabel("Number of test messages",fontsize=6.5); ax.tick_params(axis="x",labelsize=6)
ax.set_xlim(0,270)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x",color="#e3e8ee",lw=0.6); ax.set_axisbelow(True)
ax.legend(fontsize=5.5,frameon=False,loc="lower center",bbox_to_anchor=(0.4,1.0),ncol=1)
fig.tight_layout(); fig.savefig("fig4_misses_overesc.png",dpi=300,bbox_inches="tight",pad_inches=0.03)
