const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE";           // 13.3 x 7.5
p.author = "RDMU Topic 5";
p.title = "Smart Energy Grid Optimizer";

const BG="0E1117", CARD="171B26", LINE="262C3A";
const TXT="F2F4F8", MUT="8A93A6", SUB="C4CBDA";
const SOLAR="F6C445", WIND="4FC3D9", GAS="E8743B", BATT="9B8CFF";
const HEAD="Cambria", BODY="Calibri";
const W=13.3;

function bgDark(s){ s.background={color:BG}; }
function eyebrow(s,t){ s.addText(t,{x:0.6,y:0.4,w:12,h:0.3,fontFace:BODY,fontSize:11,
  color:MUT,charSpacing:5,bold:true}); }
function title(s,t,y){ s.addText(t,{x:0.6,y:y||0.75,w:12.1,h:0.9,fontFace:HEAD,
  fontSize:34,bold:true,color:TXT}); }
function card(s,x,y,w,h,fill){ s.addShape(p.shapes.ROUNDED_RECTANGLE,{x,y,w,h,
  rectRadius:0.08,fill:{color:fill||CARD},line:{color:LINE,width:1}}); }
function dot(s,x,y,c){ s.addShape(p.shapes.OVAL,{x,y,w:0.16,h:0.16,fill:{color:c}}); }

// ---------------------------------------------------------------- Slide 1
let s=p.addSlide(); bgDark(s);
s.addShape(p.shapes.RECTANGLE,{x:0,y:0,w:W,h:0.12,fill:{color:SOLAR}});
s.addText("RDMU · MAIB DSC 103 · TOPIC 5",{x:0.6,y:2.0,w:12,h:0.4,fontFace:BODY,
  fontSize:13,color:MUT,charSpacing:6,bold:true});
s.addText("Smart Energy Grid Optimizer",{x:0.6,y:2.5,w:12.1,h:1.1,fontFace:HEAD,
  fontSize:48,bold:true,color:TXT});
s.addText("Decision-making under uncertainty: balancing renewables, fossil generation\nand storage with constraint programming and reinforcement learning.",
  {x:0.6,y:3.7,w:11,h:0.9,fontFace:BODY,fontSize:17,color:SUB,lineSpacingMultiple:1.2});
[["Solar",SOLAR],["Wind",WIND],["Gas",GAS],["Battery",BATT]].forEach((it,i)=>{
  dot(s,0.6+i*1.6,4.95,it[1]);
  s.addText(it[0],{x:0.82+i*1.6,y:4.86,w:1.4,h:0.35,fontFace:BODY,fontSize:12,color:SUB});
});

// ---------------------------------------------------------------- Slide 2  Requirement
s=p.addSlide(); bgDark(s); eyebrow(s,"01 · REQUIREMENT UNDERSTANDING"); title(s,"The problem we are solving");
const reqs=[
 ["Meet demand","A smart-city grid must serve a fluctuating daily load every hour without blackouts."],
 ["Under uncertainty","Demand and renewable output are stochastic and only known through noisy forecasts."],
 ["Within constraints","Generator capacities, battery limits and a daily CO₂ cap must never be violated."],
 ["Optimise trade-offs","Minimise cost and emissions while keeping the grid reliable — a multicriteria goal."]];
reqs.forEach((r,i)=>{ const x=0.6+(i%2)*6.15, y=1.9+Math.floor(i/2)*2.0;
  card(s,x,y,5.85,1.8); dot(s,x+0.3,y+0.32,SOLAR);
  s.addText(r[0],{x:x+0.6,y:y+0.22,w:5,h:0.4,fontFace:HEAD,fontSize:18,bold:true,color:TXT});
  s.addText(r[1],{x:x+0.3,y:y+0.75,w:5.3,h:0.9,fontFace:BODY,fontSize:13.5,color:SUB,lineSpacingMultiple:1.15});
});

// ---------------------------------------------------------------- Slide 3  Concepts
s=p.addSlide(); bgDark(s); eyebrow(s,"02 · KEY RDMU CONCEPTS APPLIED"); title(s,"Four course concepts — genuinely used");
const con=[
 ["Markov Decision Process","Dispatch is modelled as state → action → reward: state = forecasts, battery charge, time, remaining carbon budget.",SOLAR],
 ["Sequential Decision Making","Battery storage couples hours together — charging now changes what is optimal later across the 24-hour horizon.",WIND],
 ["Methods for Estimation","The agent never sees the truth: it acts on noisy forecasts of demand and renewable availability.",GAS],
 ["Multicriteria Decision Making","Cost, emissions and reliability are competing objectives, resolved as a Pareto trade-off.",BATT]];
con.forEach((c,i)=>{ const y=1.85+i*1.18; card(s,0.6,y,12.1,1.05);
  s.addShape(p.shapes.OVAL,{x:0.9,y:y+0.28,w:0.5,h:0.5,fill:{color:c[2]}});
  s.addText(String(i+1),{x:0.9,y:y+0.28,w:0.5,h:0.5,align:"center",valign:"middle",fontFace:HEAD,fontSize:20,bold:true,color:BG});
  s.addText(c[0],{x:1.65,y:y+0.16,w:3.6,h:0.75,fontFace:HEAD,fontSize:17,bold:true,color:TXT,valign:"middle"});
  s.addText(c[1],{x:5.4,y:y+0.13,w:7.0,h:0.8,fontFace:BODY,fontSize:12.5,color:SUB,valign:"middle",lineSpacingMultiple:1.1});
});
s.addText("Bonus: Utility Theory — the reward function is a utility over the weighted objective.",
  {x:0.6,y:6.85,w:12,h:0.3,fontFace:BODY,fontSize:11,italic:true,color:MUT});

// ---------------------------------------------------------------- Slide 4  Flowchart
s=p.addSlide(); bgDark(s); eyebrow(s,"03 · APPLICATION FLOW CHART"); title(s,"How a decision is made each hour");
const steps=[["Noisy forecast","demand · solar · wind",WIND],["Build MDP state","+ battery, time, CO₂ left",SOLAR],
 ["Choose engine","LP solver  /  SAC agent",GAS],["Projection","enforce feasibility",BATT],
 ["Apply dispatch","update battery & emissions",WIND]];
steps.forEach((st,i)=>{ const x=0.6+i*2.5; card(s,x,2.6,2.2,1.7);
  dot(s,x+0.25,2.85,st[2]);
  s.addText(st[0],{x:x+0.2,y:3.15,w:1.85,h:0.5,fontFace:HEAD,fontSize:14,bold:true,color:TXT});
  s.addText(st[1],{x:x+0.2,y:3.65,w:1.85,h:0.55,fontFace:BODY,fontSize:10.5,color:SUB,lineSpacingMultiple:1.05});
  if(i<4) s.addText("→",{x:x+2.2,y:2.6,w:0.3,h:1.7,align:"center",valign:"middle",fontFace:BODY,fontSize:22,color:MUT});
});
s.addText("The loop repeats for all 24 hours of the simulated day; the projection step guarantees every decision is physically feasible.",
  {x:0.6,y:5.0,w:12,h:0.6,fontFace:BODY,fontSize:13,color:SUB,italic:true});

// ---------------------------------------------------------------- Slide 5  Prerequisites
s=p.addSlide(); bgDark(s); eyebrow(s,"04 · PREREQUISITES"); title(s,"What it takes to run");
const pre=[["Environment","Python 3.10+ · runs on CPU, no GPU required"],
 ["Core libraries","numpy · cvxpy (LP) · stable-baselines3 (SAC) · gymnasium"],
 ["Interface","Streamlit + Plotly dashboard; Colab notebook for reproduction"],
 ["Artifacts","Pre-trained SAC checkpoint ships in the repo for instant demo"]];
pre.forEach((r,i)=>{ const y=1.95+i*1.12; card(s,0.6,y,12.1,0.95);
  dot(s,0.95,y+0.4,[SOLAR,WIND,GAS,BATT][i]);
  s.addText(r[0],{x:1.35,y:y+0.13,w:3,h:0.7,fontFace:HEAD,fontSize:16,bold:true,color:TXT,valign:"middle"});
  s.addText(r[1],{x:4.4,y:y+0.13,w:8,h:0.7,fontFace:BODY,fontSize:13.5,color:SUB,valign:"middle"});
});

// ---------------------------------------------------------------- Slide 6  Data
s=p.addSlide(); bgDark(s); eyebrow(s,"05 · DESCRIPTION OF DATA USED"); title(s,"A calibrated synthetic grid");
s.addText([
 {text:"Why synthetic? ",options:{bold:true,color:TXT}},
 {text:"A controllable simulator lets us inject known uncertainty and test policies under disruptions a fixed dataset could not provide.",options:{color:SUB}}
],{x:0.6,y:1.85,w:12,h:0.7,fontFace:BODY,fontSize:14,lineSpacingMultiple:1.15});
const data=[["Demand","Double-peak daily curve, 80–162 MW, with Gaussian noise (the uncertainty)"],
 ["Solar","Midday bell shaped by daylight, scaled by renewable-penetration slider"],
 ["Wind","Stochastic availability across the day, partly anti-correlated with solar"],
 ["Sources","Cost ($/MWh) and emission (tCO₂/MWh) factors per generator type"]];
data.forEach((r,i)=>{ const y=2.7+i*1.02; card(s,0.6,y,12.1,0.85);
  s.addText(r[0],{x:0.95,y:y+0.1,w:2.4,h:0.65,fontFace:HEAD,fontSize:15,bold:true,color:[SOLAR,WIND,WIND,GAS][i],valign:"middle"});
  s.addText(r[1],{x:3.5,y:y+0.1,w:9,h:0.65,fontFace:BODY,fontSize:13,color:SUB,valign:"middle"});
});

// ---------------------------------------------------------------- Slide 7  Dashboard (dispatch)
s=p.addSlide(); bgDark(s); eyebrow(s,"06 · DASHBOARD"); title(s,"Live dispatch control room");
s.addImage({path:"assets/fig_dispatch.png",x:0.6,y:1.8,w:8.4,h:4.3});
card(s,9.3,1.8,3.4,4.3);
s.addText("Reading the chart",{x:9.55,y:2.0,w:3,h:0.4,fontFace:HEAD,fontSize:15,bold:true,color:TXT});
s.addText([
 {text:"Solar peaks at midday; wind carries mornings and evenings.",options:{bullet:true,breakLine:true}},
 {text:"Gas fills the residual gap; coal only when forced.",options:{bullet:true,breakLine:true}},
 {text:"Dashed line = demand. Stack meets it every hour: no blackouts.",options:{bullet:true,breakLine:true}},
 {text:"Sliders re-run the grid live: penetration, variability, carbon price.",options:{bullet:true}}
],{x:9.55,y:2.5,w:2.95,h:3.4,fontFace:BODY,fontSize:12,color:SUB,lineSpacingMultiple:1.2});

// ---------------------------------------------------------------- Slide 8  Engines compare
s=p.addSlide(); bgDark(s); eyebrow(s,"07 · TWO DECISION ENGINES"); title(s,"Constraint solver vs. learned policy");
card(s,0.6,1.95,5.95,3.9);
s.addText("Engine B — Constraint solver",{x:0.9,y:2.2,w:5.4,h:0.5,fontFace:HEAD,fontSize:18,bold:true,color:WIND});
s.addText([
 {text:"Linear program solved each hour (cvxpy).",options:{bullet:true,breakLine:true}},
 {text:"Provably optimal for the modelled problem.",options:{bullet:true,breakLine:true}},
 {text:"Never fails — the reliable baseline.",options:{bullet:true,breakLine:true}},
 {text:"Embodies Constraint Programming + Multicriteria.",options:{bullet:true}}
],{x:0.9,y:2.8,w:5.4,h:2.8,fontFace:BODY,fontSize:13,color:SUB,lineSpacingMultiple:1.3});
card(s,6.75,1.95,5.95,3.9);
s.addText("Engine A — SAC + projection",{x:7.05,y:2.2,w:5.4,h:0.5,fontFace:HEAD,fontSize:18,bold:true,color:BATT});
s.addText([
 {text:"Soft Actor-Critic learns a dispatch policy.",options:{bullet:true,breakLine:true}},
 {text:"Entropy-driven exploration vs. exploitation.",options:{bullet:true,breakLine:true}},
 {text:"A projection layer forces every action feasible.",options:{bullet:true,breakLine:true}},
 {text:"\"Safe RL\" — adapts to dynamics the LP cannot model.",options:{bullet:true}}
],{x:7.05,y:2.8,w:5.4,h:2.8,fontFace:BODY,fontSize:13,color:SUB,lineSpacingMultiple:1.3});
s.addText("The LP is the optimal benchmark; the agent trades a little optimality for adaptability — and the projection keeps it safe.",
  {x:0.6,y:6.1,w:12,h:0.5,fontFace:BODY,fontSize:12.5,italic:true,color:MUT});

// ---------------------------------------------------------------- Slide 9  Pareto / sensitivity
s=p.addSlide(); bgDark(s); eyebrow(s,"08 · SENSITIVITY ANALYSIS"); title(s,"The cost–emissions trade-off");
s.addImage({path:"assets/fig_pareto.png",x:0.6,y:1.8,w:8.4,h:4.3});
card(s,9.3,1.8,3.4,4.3);
s.addText("Multicriteria insight",{x:9.55,y:2.0,w:3,h:0.4,fontFace:HEAD,fontSize:15,bold:true,color:TXT});
s.addText([
 {text:"Each point is a different carbon price ($/tCO₂).",options:{bullet:true,breakLine:true}},
 {text:"Raising the price cuts emissions sharply at first…",options:{bullet:true,breakLine:true}},
 {text:"…then flattens — the renewable/storage limit binds.",options:{bullet:true,breakLine:true}},
 {text:"This frontier is the heart of the decision under uncertainty.",options:{bullet:true}}
],{x:9.55,y:2.5,w:2.95,h:3.4,fontFace:BODY,fontSize:12,color:SUB,lineSpacingMultiple:1.2});

// ---------------------------------------------------------------- Slide 10  Outcomes
s=p.addSlide(); bgDark(s);
s.addShape(p.shapes.RECTANGLE,{x:0,y:0,w:W,h:0.12,fill:{color:SOLAR}});
eyebrow(s,"09 · OUTCOMES & METRICS"); title(s,"What the module delivers");
const kpi=[["63%","Renewable share at base settings",SOLAR],["0 MWh","Unmet demand — full reliability",WIND],
 ["~50%","Emissions cut as carbon price rises",GAS],["24 h","Sequential horizon optimised",BATT]];
kpi.forEach((k,i)=>{ const x=0.6+i*3.05; card(s,x,2.0,2.85,1.9);
  s.addText(k[0],{x:x+0.2,y:2.2,w:2.5,h:0.8,fontFace:HEAD,fontSize:34,bold:true,color:k[2]});
  s.addText(k[1],{x:x+0.2,y:3.05,w:2.5,h:0.75,fontFace:BODY,fontSize:11.5,color:SUB,lineSpacingMultiple:1.1});
});
s.addText("Deliverables: stochastic grid simulator · LP + SAC engines · interactive Streamlit dashboard · Pareto sensitivity · reproducible Colab notebook.",
  {x:0.6,y:4.3,w:12,h:0.8,fontFace:BODY,fontSize:13.5,color:SUB,lineSpacingMultiple:1.2});
s.addText("Concepts demonstrated: Markov Decision Process · Sequential Decision Making · Methods for Estimation · Multicriteria Decision Making (+ Utility Theory).",
  {x:0.6,y:5.2,w:12,h:0.8,fontFace:BODY,fontSize:13,italic:true,color:MUT,lineSpacingMultiple:1.2});

p.writeFile({fileName:"SmartGrid_RDMU_Topic5.pptx"}).then(f=>console.log("WROTE",f));
