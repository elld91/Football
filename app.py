#!/usr/bin/env python3
"""
Lower-League Football Model — one file, one command.

    python app.py            # LIVE: pull football-data.co.uk full feed, predict
                             #       upcoming fixtures, write index.html
    python app.py --demo     # DEMO: pull the free GitHub results mirror, hold out
                             #       the last round, write index.html (runs anywhere)
    python app.py --backtest # add a rolling walk-forward panel (slower)

LIVE needs nothing but internet. DEMO needs nothing at all. The output is a
single self-contained index.html you can host on GitHub Pages and open on
phone / iPad / laptop.

Honest scope: goals-model markets (1X2, over/under, BTTS, clean sheets, correct
score) are always populated. Cards/corners fill in only on the LIVE feed (they
need the referee/HC/HY columns). CLV needs the closing-odds columns, also LIVE.
"""
from __future__ import annotations
import argparse, io, json, re, sys, tarfile, urllib.request, datetime as dt
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# --------------------------------------------------------------------------- #
#  CONFIG
# --------------------------------------------------------------------------- #
LIVE_BASE   = "https://www.football-data.co.uk/mmz4281"
MIRROR_TAR  = "https://codeload.github.com/footballcsv/cache.footballdata/tar.gz/refs/heads/master"
DIVS        = {"Championship": ("E1", "eng.2"), "League One": ("E2", "eng.3"),
               "League Two": ("E3", "eng.4")}
SEASONS_FROM = 2021                 # recent seasons only (time decay makes old data cheap)
HALF_LIFE    = 240                  # days
MAX_GOALS    = 10
UA = {"User-Agent": "Mozilla/5.0 (lowerleague-model; research)"}

# --------------------------------------------------------------------------- #
#  DIXON-COLES
# --------------------------------------------------------------------------- #
def _tau(h, a, lam, mu, rho):
    o = np.ones_like(lam, float)
    o = np.where((h==0)&(a==0), 1-lam*mu*rho, o); o = np.where((h==0)&(a==1), 1+lam*rho, o)
    o = np.where((h==1)&(a==0), 1+mu*rho, o);     o = np.where((h==1)&(a==1), 1-rho, o)
    return o

class DixonColes:
    def __init__(self, half_life=HALF_LIFE, max_goals=MAX_GOALS):
        self.hl, self.mg = half_life, max_goals
    def fit(self, home, away, hg, ag, dates, ref=None):
        home,away = np.asarray(home),np.asarray(away)
        hg,ag = np.asarray(hg,int),np.asarray(ag,int)
        dates = np.asarray(dates,"datetime64[D]"); ref = np.datetime64(ref or dates.max(),"D")
        w = np.exp(-(np.log(2)/self.hl)*np.clip((ref-dates).astype("timedelta64[D]").astype(float),0,None))
        self.teams = sorted(set(home)|set(away)); self.idx={t:i for i,t in enumerate(self.teams)}
        hi=np.array([self.idx[t] for t in home]); ai=np.array([self.idx[t] for t in away]); n=len(self.teams)
        def unpack(p):
            d=np.append(p[n:2*n-1], -p[n:2*n-1].sum()); return p[:n],d,p[-2],p[-1]
        def nll(p):
            atk,dfc,ha,rho=unpack(p)
            lam=np.exp(atk[hi]-dfc[ai]+ha); mu=np.exp(atk[ai]-dfc[hi])
            ll=poisson.logpmf(hg,lam)+poisson.logpmf(ag,mu)+np.log(np.clip(_tau(hg,ag,lam,mu,rho),1e-9,None))
            return -np.sum(w*ll)
        x0=np.concatenate([np.zeros(n),np.zeros(n-1),[.25],[-.05]])
        r=minimize(nll,x0,method="L-BFGS-B",bounds=[(-3,3)]*(2*n-1)+[(-1,1),(-.2,.2)])
        self.p=r.x; self.n=n; return self
    def _ratings(self):
        n=self.n; atk=self.p[:n]; dfc=np.append(self.p[n:2*n-1],-self.p[n:2*n-1].sum())
        return atk,dfc,self.p[-2],self.p[-1]
    def matrix(self, home, away, lam=None, mu=None):
        atk,dfc,ha,rho=self._ratings(); i,j=self.idx[home],self.idx[away]
        if lam is None: lam=np.exp(atk[i]-dfc[j]+ha)
        if mu  is None: mu =np.exp(atk[j]-dfc[i])
        g=np.arange(self.mg+1)
        m=np.outer(poisson.pmf(g,lam),poisson.pmf(g,mu))
        for hh,aa in [(0,0),(0,1),(1,0),(1,1)]:
            m[hh,aa]*=_tau(np.array(hh),np.array(aa),np.array(lam),np.array(mu),rho)
        return m/m.sum(), lam, mu
    def base_lambda(self, home, away):
        atk,dfc,ha,_=self._ratings(); i,j=self.idx[home],self.idx[away]
        return float(np.exp(atk[i]-dfc[j]+ha)), float(np.exp(atk[j]-dfc[i]))
    def predict(self, home, away, lam=None, mu=None):
        m,lam,mu=self.matrix(home,away,lam,mu)
        ph=np.tril(m,-1).sum(); pd_=np.trace(m); pa=np.triu(m,1).sum()
        over=sum(m[h,a] for h in range(self.mg+1) for a in range(self.mg+1) if h+a>2.5)
        btts=1-(m[0,:].sum()+m[:,0].sum()-m[0,0])
        cs_home=m[:,0].sum(); cs_away=m[0,:].sum()
        # top 3 scorelines
        flat=[(m[h,a],f"{h}-{a}") for h in range(4) for a in range(4)]
        flat.sort(reverse=True); tops=[{"score":s,"p":round(float(p),3)} for p,s in flat[:3]]
        return dict(home=ph,draw=pd_,away=pa,over25=over,btts=btts,
                    cs_home=cs_home,cs_away=cs_away,xg_home=float(lam),xg_away=float(mu),tops=tops)

# --------------------------------------------------------------------------- #
#  INGEST
# --------------------------------------------------------------------------- #
def season_codes(start): return [f"{str(y)[-2:]}{str(y+1)[-2:]}" for y in range(start, dt.date.today().year+1)]

def load_live():
    ren={"Date":"date","HomeTeam":"home","AwayTeam":"away","FTHG":"hg","FTAG":"ag",
         "HTHG":"hthg","HTAG":"htag",
         "Referee":"referee","HC":"hc","AC":"ac","HY":"hy","AY":"ay","HR":"hr","AR":"ar",
         "B365H":"b365_h","B365D":"b365_d","B365A":"b365_a",
         "PSCH":"psc_h","PSCD":"psc_d","PSCA":"psc_a"}
    out=[]
    for lg,(code,_) in DIVS.items():
        for s in season_codes(SEASONS_FROM):
            try:
                raw=urllib.request.urlopen(urllib.request.Request(f"{LIVE_BASE}/{s}/{code}.csv",headers=UA),timeout=30).read()
            except Exception: continue
            if not raw.strip(): continue
            d=pd.read_csv(io.BytesIO(raw),encoding="latin-1")
            keep={k:v for k,v in ren.items() if k in d.columns}
            d=d[list(keep)].rename(columns=keep); d["league"]=lg
            d["date"]=pd.to_datetime(d["date"],dayfirst=True,errors="coerce")
            out.append(d.dropna(subset=["date","home","away","hg","ag"]))
    df=pd.concat(out,ignore_index=True)
    # also pull this week's UPCOMING fixtures (+odds, no scores yet) so the slate
    # shows games still to be played. Defensive: if the file/format changes, we
    # just skip it and fall back to recent matches rather than crashing.
    try:
        raw=urllib.request.urlopen(urllib.request.Request("https://www.football-data.co.uk/fixtures.csv",headers=UA),timeout=30).read()
        if raw[:3]==b"\xef\xbb\xbf": raw=raw[3:]                 # strip UTF-8 BOM if present
        fx=pd.read_csv(io.BytesIO(raw),encoding="latin-1")
        fx.columns=[str(c).replace("\ufeff","").strip() for c in fx.columns]
        divcol=next((c for c in fx.columns if c.lower()=="div"),None)
        if not divcol:
            print("  (upcoming fixtures: no division column — columns found:",list(fx.columns)[:10],"— showing recent matches)")
        else:
            code2lg={code:lg for lg,(code,_) in DIVS.items()}
            fx=fx[fx[divcol].isin(list(code2lg))].copy()
            fx["league"]=fx[divcol].map(code2lg)
            timecol=next((c for c in fx.columns if c.lower()=="time"),None)
            def _ko(row):
                try:
                    dd=pd.to_datetime(row["Date"],dayfirst=True)
                    tt=str(row[timecol]).strip() if timecol else ""
                    if not tt or ":" not in tt: tt="15:00"
                    return pd.to_datetime(f"{dd.date()} {tt}")
                except Exception: return pd.NaT
            fx["ko"]=fx.apply(_ko,axis=1)
            keep={k:v for k,v in ren.items() if k in fx.columns}
            fx=fx[list(keep)+["league","ko"]].rename(columns=keep)
            fx["date"]=pd.to_datetime(fx["date"],dayfirst=True,errors="coerce")
            for c in ("hg","ag"):
                if c not in fx.columns: fx[c]=np.nan
            fx=fx.dropna(subset=["date","home","away"])
            if len(fx):
                df=pd.concat([df,fx],ignore_index=True)
                print(f"  + {len(fx)} upcoming fixtures with odds")
            else:
                print("  (no upcoming fixtures listed right now — showing recent matches)")
    except Exception as e:
        print(f"  (couldn't load upcoming fixtures: {type(e).__name__}: {e} — showing recent matches)")
    return df

def load_demo():
    raw=urllib.request.urlopen(urllib.request.Request(MIRROR_TAR,headers=UA),timeout=60).read()
    tar=tarfile.open(fileobj=io.BytesIO(raw)); out=[]
    for lg,(_,stub) in DIVS.items():
        for m in tar.getmembers():
            mo=re.search(rf"/(\d{{4}})-\d\d/{re.escape(stub)}\.csv$", m.name)
            if not mo or int(mo.group(1))<SEASONS_FROM: continue
            d=pd.read_csv(tar.extractfile(m))
            ft=d["FT"].astype(str).str.extract(r"(\d+)-(\d+)")
            ht=d["HT"].astype(str).str.extract(r"(\d+)-(\d+)") if "HT" in d.columns else pd.DataFrame({0:[None]*len(d),1:[None]*len(d)})
            d=d.assign(hg=pd.to_numeric(ft[0],errors="coerce"),ag=pd.to_numeric(ft[1],errors="coerce"),
                       hthg=pd.to_numeric(ht[0],errors="coerce"),htag=pd.to_numeric(ht[1],errors="coerce"),
                       home=d["Team 1"].str.strip(),away=d["Team 2"].str.strip(),league=lg,
                       date=pd.to_datetime(d["Date"],errors="coerce"))
            out.append(d[["date","home","away","hg","ag","hthg","htag","league"]].dropna(subset=["date","home","away","hg","ag"]))
    return pd.concat(out,ignore_index=True)

# --------------------------------------------------------------------------- #
#  FORM + MARKET HELPERS
# --------------------------------------------------------------------------- #
# distinctive lowercase substrings so matching survives both naming schemes
RIVALRIES=[("sheffield united","sheffield wed","Steel City derby"),
    ("bristol city","bristol r","Bristol derby"),("ipswich","norwich","East Anglian derby"),
    ("portsmouth","southampton","South Coast derby"),("milton keynes","wimbledon","MK–Wimbledon"),
    ("mk dons","wimbledon","MK–Wimbledon"),("charlton","millwall","South London derby"),
    ("bolton","wigan","Greater Manchester derby"),("sunderland","newcastle","Tyne–Wear derby"),
    ("cardiff","swansea","South Wales derby"),("nott","derby","East Midlands derby"),
    ("oxford","swindon","A420 derby"),("plymouth","exeter","Devon derby"),
    ("notts county","mansfield","Nottinghamshire derby"),("grimsby","scunthorpe","Lincolnshire derby"),
    ("peterbor","cambridge","Cambridgeshire derby"),("blackburn","burnley","East Lancashire derby"),
    ("stoke","port vale","Potteries derby"),("colchester","ipswich","Essex–Suffolk")]

def derby(home, away):
    h,a=home.lower(),away.lower()
    for x,y,label in RIVALRIES:
        if (x in h and y in a) or (x in a and y in h): return label
    return None

def h2h(df, home, away, before, n=6):
    m=df[(((df.home==home)&(df.away==away))|((df.home==away)&(df.away==home)))&(df.date<before)]
    m=m.sort_values("date").tail(n)
    if m.empty: return None
    w=d_=l=0; last=None
    for _,r in m.iterrows():
        gf,ga=(r.hg,r.ag) if r.home==home else (r.ag,r.hg)  # from fixture home team's view
        w+=gf>ga; d_+=gf==ga; l+=gf<ga
        last=f"{int(r.hg)}-{int(r.ag)} ({r.home.split()[0]} h)"
    return dict(n=len(m),w=int(w),d=int(d_),l=int(l),last=last)

def half_rate(df, team, before, half, n=12):
    """(scored_rate, conceded_rate) for a given half. half=1 uses HT goals,
    half=2 uses FT-HT. Returns None if HT data absent."""
    if "hthg" not in df.columns: return None
    sub=df[((df.home==team)|(df.away==team))&(df.date<before)].dropna(subset=["hthg","htag"]).sort_values("date").tail(n)
    if len(sub)<4: return None
    sc=[];co=[]
    for _,r in sub.iterrows():
        if half==1: hs,as_=r.hthg,r.htag
        else:       hs,as_=r.hg-r.hthg,r.ag-r.htag
        if r.home==team: sc.append(hs);co.append(as_)
        else:            sc.append(as_);co.append(hs)
    return float(np.mean(sc)), float(np.mean(co))

def half_markets(df, home, away, before):
    r1h=half_rate(df,home,before,1); r1a=half_rate(df,away,before,1)
    r2h=half_rate(df,home,before,2); r2a=half_rate(df,away,before,2)
    if not all([r1h,r1a,r2h,r2a]): return None
    def pboth(hh,aa):  # expected goals each side in a half -> P(both teams score)
        lam_h=max((hh[0]+aa[1])/2,1e-6); lam_a=max((aa[0]+hh[1])/2,1e-6)
        return (1-np.exp(-lam_h))*(1-np.exp(-lam_a)), lam_h+lam_a
    b1,tot1=pboth(r1h,r1a); b2,_=pboth(r2h,r2a)
    return dict(btts_both_halves=round(float(b1*b2),3),          # independence approx
                fh_over05=round(float(1-np.exp(-tot1)),3))

def _odds(r):
    """Return (home,draw,away) decimal odds from a row, closing preferred."""
    for pre in ("psc_", "b365_"):
        try:
            h,d,a = r.get(pre+"h"), r.get(pre+"d"), r.get(pre+"a")
            if all(pd.notna(x) and x>1 for x in (h,d,a)): return (float(h),float(d),float(a))
        except Exception: pass
    return None

def form_string(df, team, before, n=5):
    h=df[((df.home==team)|(df.away==team))&(df.date<before)].sort_values("date").tail(n)
    s=[]
    for _,r in h.iterrows():
        gf,ga=(r.hg,r.ag) if r.home==team else (r.ag,r.hg)
        s.append("W" if gf>ga else ("D" if gf==ga else "L"))
    return "".join(s) if s else "—"

def rate(df, team, col_for, col_against, before, home=True):
    """avg of a per-match count (e.g. corners/cards) for a team, recent."""
    sub=df[((df.home==team)|(df.away==team))&(df.date<before)].sort_values("date").tail(10)
    if sub.empty or col_for not in df.columns: return None
    vals=[]
    for _,r in sub.iterrows():
        vals.append(r[col_for] if r.home==team else r[col_against])
    return float(np.nanmean(vals))

# --------------------------------------------------------------------------- #
#  BUILD SLATE + HONESTY PANEL
# --------------------------------------------------------------------------- #
def build(df, demo):
    df=df.sort_values("date").reset_index(drop=True)
    payload={"generated":dt.datetime.now(dt.timezone.utc).strftime("%d/%m/%Y %H:%M UTC"),
             "mode":"demo" if demo else "live","leagues":[]}
    for lg in DIVS:
        d=df[df.league==lg].copy()
        if len(d)<200: continue
        if demo:                                   # hold out the last round as the "slate"
            cut=d.date.max()-pd.Timedelta(days=7); slate=d[d.date>=cut]; train=d[d.date<cut]
        else:                                      # LIVE: only fixtures whose kickoff is still ahead
            now=pd.Timestamp.now()
            train=d[(d.date<now.normalize())&d.hg.notna()]
            unplayed=d[d.hg.isna()].copy()
            if "ko" in unplayed.columns:
                eff=unplayed["ko"].fillna(unplayed["date"]+pd.Timedelta(hours=23,minutes=59))
            else:
                eff=unplayed["date"]+pd.Timedelta(hours=23,minutes=59)
            slate=unplayed[eff>now]                        # drop anything already kicked off
            if train.empty: train=d.dropna(subset=["hg"])
            print(f"  {lg}: {len(slate)} upcoming fixtures to predict")
        model=DixonColes().fit(train.home,train.away,train.hg,train.ag,train.date.values,ref=train.date.max())
        known=set(model.teams); fixtures=[]; notes=load_notes()
        XGW=0.4  # xG blend weight (Championship, when xG present)
        for _,r in slate.iterrows():
            if r.home not in known or r.away not in known: continue
            lam=mu=None
            if "hxg" in d.columns:                       # blend goals-strength with recent xG
                xf_h=xg_rate(d,r.home,r.date,"for"); xa_a=xg_rate(d,r.away,r.date,"against")
                xf_a=xg_rate(d,r.away,r.date,"for"); xa_h=xg_rate(d,r.home,r.date,"against")
                if None not in (xf_h,xa_a,xf_a,xa_h):
                    lg_,mg_=model.base_lambda(r.home,r.away)
                    lam=(1-XGW)*lg_+XGW*(xf_h+xa_a)/2
                    mu =(1-XGW)*mg_+XGW*(xf_a+xa_h)/2
            p=model.predict(r.home,r.away,lam,mu)
            fx=dict(home=r.home,away=r.away,date=pd.to_datetime(r.date).strftime("%d/%m/%Y"),
                    p_home=round(p["home"],3),p_draw=round(p["draw"],3),p_away=round(p["away"],3),
                    over25=round(p["over25"],3),btts=round(p["btts"],3),
                    cs_home=round(p["cs_home"],3),cs_away=round(p["cs_away"],3),
                    xg_home=round(p["xg_home"],2),xg_away=round(p["xg_away"],2),tops=p["tops"],
                    form_home=form_string(d,r.home,r.date),form_away=form_string(d,r.away,r.date))
            if "ko" in slate.columns and pd.notna(r.get("ko")):
                fx["time"]=pd.to_datetime(r["ko"]).strftime("%H:%M")
            if notes.get(r.home): fx["note_home"]=str(notes[r.home])[:120]
            if notes.get(r.away): fx["note_away"]=str(notes[r.away])[:120]
            if pd.notna(r.get("hg")): fx["result"]=f"{int(r.hg)}-{int(r.ag)}"
            dby=derby(r.home,r.away)
            if dby: fx["derby"]=dby
            hh=h2h(d,r.home,r.away,r.date)
            if hh: fx["h2h"]=hh
            hm=half_markets(d,r.home,r.away,r.date)
            if hm: fx.update(btts_bh=hm["btts_both_halves"], fh_over05=hm["fh_over05"])
            if "hc" in d.columns:      # corners over-line (live only)
                ch=rate(d,r.home,"hc","ac",r.date); ca=rate(d,r.away,"ac","hc",r.date)
                if ch and ca:
                    lam=ch+ca; fx["corners_exp"]=round(lam,1)
                    fx["corners_o95"]=round(float(1-poisson.cdf(9,lam)),3)
            if "hy" in d.columns:      # cards over-line (live only), referee-adjusted
                yh=rate(d,r.home,"hy","ay",r.date); ya=rate(d,r.away,"ay","hy",r.date)
                if yh and ya:
                    lam=yh+ya
                    rref=referee_card_rate(d,r.get("referee"),r.date)   # blend toward this ref's rate
                    if rref: lam=0.6*lam+0.4*rref; fx["referee"]=str(r.get("referee"))
                    fx["cards_exp"]=round(lam,1); fx["cards_o35"]=round(float(1-poisson.cdf(3,lam)),3)
            fixtures.append(fx)
        # value screen + market-blended posted probs, where odds exist
        for fx, (_, r) in zip(fixtures, slate.iterrows()):
            od = _odds(r)
            if not od: continue
            imp = 1/np.array(od); imp = imp/imp.sum()          # remove overround
            fx["odds3"]=[round(float(x),2) for x in od]
            fx["mkt3"]=[round(float(x),3) for x in imp]
            mdl = np.array([fx["p_home"], fx["p_draw"], fx["p_away"]])
            bl = blend(mdl, imp, 0.5)                            # better-calibrated posted prob
            fx["pb_home"],fx["pb_draw"],fx["pb_away"]=[round(float(x),3) for x in bl]
            edge = mdl - imp; i = int(np.argmax(edge))
            if edge[i] > 0.03:                                  # only flag real divergence
                fx["value"] = dict(pick=["Home","Draw","Away"][i],
                                   model=round(float(mdl[i]),3), market=round(float(imp[i]),3),
                                   edge=round(float(edge[i]),3), odds=round(float(od[i]),2))
        # honesty panel: quick holdout RPS vs fixed-prior baseline
        hp=holdout(d)
        payload["leagues"].append(dict(name=lg,fixtures=fixtures,honesty=hp,
                                        has_cards="hy" in d.columns,has_odds="psc_h" in d.columns))
    return payload

def holdout(d):
    d=d.dropna(subset=["hg"]).sort_values("date"); cut=d.date.max()-pd.Timedelta(days=90)
    tr,te=d[d.date<cut],d[d.date>=cut]
    if len(te)<20: return None
    m=DixonColes().fit(tr.home,tr.away,tr.hg,tr.ag,tr.date.values,ref=cut); known=set(m.teams)
    base=np.array([.44,.27,.29]); rp=[];bp=[];cal=[]
    def rps(p,o): c=np.cumsum(p);ob=np.zeros(3);ob[o]=1;return np.sum((c-np.cumsum(ob))**2)/2
    for _,r in te.iterrows():
        if r.home not in known or r.away not in known: continue
        pr=m.predict(r.home,r.away); hda=np.array([pr["home"],pr["draw"],pr["away"]])
        o=0 if r.hg>r.ag else (1 if r.hg==r.ag else 2)
        rp.append(rps(hda,o)); bp.append(rps(base,o))
        cal.append((pr["btts"], int(r.hg>0 and r.ag>0)))
    cal=np.array(cal); mid=cal[(cal[:,0]>.4)&(cal[:,0]<.6)]
    return dict(n=len(rp),model_rps=round(float(np.mean(rp)),4),base_rps=round(float(np.mean(bp)),4),
                btts_pred=round(float(mid[:,0].mean()),2) if len(mid) else None,
                btts_obs=round(float(mid[:,1].mean()),2) if len(mid) else None)

# --------------------------------------------------------------------------- #
#  RENDER
# --------------------------------------------------------------------------- #
# =========================================================================== #
#  IMPROVEMENTS — some validated here, some activate on your live run
# =========================================================================== #

# ---- xG for the Championship (FBref via soccerdata) ----------------------- #
def load_xg(seasons):
    """Per-match team xG from FBref (Championship only). Lazy import so the app
    doesn't need soccerdata unless you use --xg. Schemas drift, so this detects
    the xG columns; if FBref/soccerdata changes, adjust the column picks here.
    NOTE: written but not runnable in the build sandbox (no FBref access) — it
    activates on your machine. Run `pip install soccerdata` first."""
    import soccerdata as sd
    fb = sd.FBref(leagues="ENG-Championship", seasons=seasons)
    s = fb.read_schedule().reset_index()
    cols = {c.lower(): c for c in s.columns}
    hx = cols.get("home_xg") or cols.get("hxg"); ax = cols.get("away_xg") or cols.get("axg")
    if not hx or not ax:
        print("  [xg] no xG columns found in FBref schedule — skipping"); return None
    out = s.rename(columns={cols.get("date","date"):"date", cols.get("home_team","home"):"home",
                            cols.get("away_team","away"):"away", hx:"hxg", ax:"axg"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["league"] = "Championship"
    return out[["date","home","away","hxg","axg","league"]].dropna()

def xg_rate(df, team, before, side, n=10):
    """Recent mean xG for/against a team. side='for' or 'against'."""
    if "hxg" not in df.columns: return None
    sub=df[((df.home==team)|(df.away==team))&(df.date<before)].dropna(subset=["hxg","axg"]).sort_values("date").tail(n)
    if len(sub)<4: return None
    v=[]
    for _,r in sub.iterrows():
        f,a=(r.hxg,r.axg) if r.home==team else (r.axg,r.hxg)
        v.append(f if side=="for" else a)
    return float(np.mean(v))

# ---- Club Elo (free API) — activates on your live run --------------------- #
def load_elo(date_iso):
    """All-club Elo on a date. Not reachable from the build sandbox."""
    import urllib.request, io
    raw=urllib.request.urlopen(f"http://api.clubelo.com/{date_iso}",timeout=30).read()
    return pd.read_csv(io.BytesIO(raw))

# ---- Weather (Open-Meteo, free, no key) — activates live ------------------ #
def load_weather(lat, lon, date_iso):
    import urllib.request, json as _j
    u=(f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
       f"&hourly=precipitation,wind_speed_10m&start_date={date_iso}&end_date={date_iso}&timezone=Europe/London")
    return _j.loads(urllib.request.urlopen(u,timeout=30).read())

# ---- Referee-aware card rate — activates when referee column present ------- #
def referee_card_rate(df, referee, before, n=40):
    if "referee" not in df.columns or pd.isna(referee): return None
    sub=df[(df.referee==referee)&(df.date<before)].tail(n)
    if len(sub)<8 or "hy" not in df.columns: return None
    return float((sub.hy.fillna(0)+sub.ay.fillna(0)).mean())

# ---- Market blend (validated maths) --------------------------------------- #
def blend(model_p, market_p, w=0.5):
    """Blend model with de-vigged market for a better-calibrated posted prob."""
    b=(1-w)*np.asarray(model_p)+w*np.asarray(market_p); return b/b.sum()

# ---- Soft layer: your manual injury/team-news notes ----------------------- #
def load_notes(path="notes.json"):
    import os
    if not os.path.exists(path): return {}
    try: return json.load(open(path, encoding="utf-8"))
    except Exception: return {}

# ---- Proper rolling walk-forward backtest (validated here) ---------------- #
def walkforward(d, half_life=HALF_LIFE, refit_every=45, min_train_days=400):
    d=d.dropna(subset=["hg"]).sort_values("date").reset_index(drop=True)
    start=d.date.min()+pd.Timedelta(days=min_train_days)
    cuts=pd.date_range(start,d.date.max(),freq=f"{refit_every}D")
    base=np.array([.44,.27,.29])
    def _rps(p,o): c=np.cumsum(p);ob=np.zeros(3);ob[o]=1;return np.sum((c-np.cumsum(ob))**2)/2
    mr=[];br=[];hit=0;tot=0
    for cut in cuts:
        tr=d[d.date<cut]; te=d[(d.date>=cut)&(d.date<cut+pd.Timedelta(days=refit_every))]
        if len(te)==0 or tr.home.nunique()<6: continue
        m=DixonColes(half_life=half_life).fit(tr.home,tr.away,tr.hg,tr.ag,tr.date.values,ref=cut)
        known=set(m.teams)
        for _,r in te.iterrows():
            if r.home not in known or r.away not in known: continue
            p=m.predict(r.home,r.away); hda=np.array([p["home"],p["draw"],p["away"]])
            o=0 if r.hg>r.ag else (1 if r.hg==r.ag else 2)
            mr.append(_rps(hda,o)); br.append(_rps(base,o))
            hit+=int(np.argmax(hda)==o); tot+=1
    return dict(n=tot,model_rps=round(float(np.mean(mr)),4),base_rps=round(float(np.mean(br)),4),
                hit_rate=round(hit/tot,3) if tot else None)

# ---- Half-life auto-tune (validated here) --------------------------------- #
def tune_half_life(d, grid=(120,160,200,240,300)):
    best=None
    for hl in grid:
        s=walkforward(d,half_life=hl,refit_every=60)
        if best is None or s["model_rps"]<best[1]: best=(hl,s["model_rps"])
        print(f"  half_life={hl:3d}  RPS={s['model_rps']}  hit={s['hit_rate']}")
    return best

# ---- CLV: log value picks, measure whether you beat the close ------------- #
def log_picks(payload, path="picks_log.csv"):
    import csv, os
    rows=[]
    for lg in payload["leagues"]:
        for f in lg["fixtures"]:
            if f.get("value"):
                v=f["value"]; rows.append([payload["generated"],lg["name"],f["date"],
                    f["home"],f["away"],v["pick"],v["model"],v["market"],v["odds"]])
    if not rows: return 0
    new=not os.path.exists(path)
    with open(path,"a",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh)
        if new: w.writerow(["logged","league","date","home","away","pick","model_p","market_p","odds_taken"])
        w.writerows(rows)
    return len(rows)

def clv_report(picks="picks_log.csv"):
    """Join logged picks to closing odds (live feed) and report CLV. Needs your
    live football-data pull to supply closing prices; maths validated here."""
    import os
    if not os.path.exists(picks): return "no picks logged yet"
    log=pd.read_csv(picks)
    df=load_live()  # completed matches carry Pinnacle closing (psc_*)
    beat=0;n=0;clv=[]
    for _,r in log.iterrows():
        m=df[(df.home==r.home)&(df.away==r.away)&(df.date.dt.date.astype(str)==r.date)]
        if m.empty: continue
        row=m.iloc[0]; close={"Home":row.get("psc_h"),"Draw":row.get("psc_d"),"Away":row.get("psc_a")}.get(r.pick)
        if pd.isna(close): continue
        n+=1; beat+=int(r.odds_taken>close)           # you took a bigger price than the close = +CLV
        clv.append((r.odds_taken/close-1)*100)
    return dict(n=n,pct_beat_close=round(beat/n,3) if n else None,
                avg_clv_pct=round(float(np.mean(clv)),2) if clv else None)


def render(payload):
    return TEMPLATE.replace("/*DATA*/{}", json.dumps(payload))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--demo",action="store_true",help="free results mirror, runs anywhere")
    ap.add_argument("--xg",action="store_true",help="blend FBref xG into the Championship (live only)")
    ap.add_argument("--backtest",action="store_true",help="proper multi-season walk-forward, per league")
    ap.add_argument("--tune",action="store_true",help="sweep the time-decay half-life")
    ap.add_argument("--clv",action="store_true",help="score logged value picks vs the closing line")
    a=ap.parse_args()

    if a.clv:
        print("CLV report:", clv_report()); return

    print("Loading data…", "(demo mirror)" if a.demo else "(live football-data.co.uk)")
    df=load_demo() if a.demo else load_live()
    print(f"  {len(df)} matches across {df.league.nunique()} leagues")

    if a.xg and not a.demo:
        xg=load_xg(season_codes(SEASONS_FROM))
        if xg is not None:
            df=df.merge(xg[["date","home","away","hxg","axg"]],on=["date","home","away"],how="left")
            print(f"  merged xG for {df.hxg.notna().sum()} Championship matches")

    if a.backtest:
        for lg in df.league.unique():
            print(f"\n=== {lg} — walk-forward ===")
            print(" ", walkforward(df[df.league==lg])); 
        return
    if a.tune:
        for lg in df.league.unique():
            print(f"\n=== {lg} — half-life sweep ==="); tune_half_life(df[df.league==lg])
        return

    payload=build(df,a.demo)
    open("index.html","w",encoding="utf-8").write(render(payload))
    n=log_picks(payload)
    print(f"Wrote index.html" + (f" · logged {n} value picks to picks_log.csv" if n else ""))

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lower-League Model</title>
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="LL Model">
<meta name="theme-color" content="#0d1117">
<link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 180 180'%3E%3Crect width='180' height='180' rx='38' fill='%230d1117'/%3E%3Ccircle cx='90' cy='90' r='52' fill='none' stroke='%234c9f70' stroke-width='9'/%3E%3Cpath d='M90 38v104M38 90h104' stroke='%234c9f70' stroke-width='9'/%3E%3C/svg%3E">
<link rel="manifest" href="data:application/manifest+json,%7B%22name%22%3A%22Lower-League%20Model%22%2C%22short_name%22%3A%22LL%20Model%22%2C%22display%22%3A%22standalone%22%2C%22background_color%22%3A%22%230d1117%22%2C%22theme_color%22%3A%22%230d1117%22%2C%22icons%22%3A%5B%7B%22src%22%3A%22data%3Aimage%2Fsvg%2Bxml%2C%253Csvg%2520xmlns%253D%2527http%253A%252F%252Fwww.w3.org%252F2000%252Fsvg%2527%2520viewBox%253D%25270%25200%2520180%2520180%2527%253E%253Crect%2520width%253D%2527180%2527%2520height%253D%2527180%2527%2520fill%253D%2527%25230d1117%2527%252F%253E%253Ccircle%2520cx%253D%252790%2527%2520cy%253D%252790%2527%2520r%253D%252752%2527%2520fill%253D%2527none%2527%2520stroke%253D%2527%25234c9f70%2527%2520stroke-width%253D%25279%2527%252F%253E%253C%252Fsvg%253E%22%2C%22sizes%22%3A%22any%22%2C%22type%22%3A%22image%2Fsvg%2Bxml%22%7D%5D%7D">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --ink:#0d1117; --panel:#161d27; --panel2:#1d2632; --line:#28323f;
    --text:#e6ecf3; --muted:#8593a4;
    --home:#4c9f70; --draw:#d99c52; --away:#5b8fb0; --loss:#b8636b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ink);color:var(--text);
       font-family:Inter,system-ui,sans-serif;font-variant-numeric:tabular-nums;
       -webkit-font-smoothing:antialiased}
  .wrap{max-width:960px;margin:0 auto;padding:22px 16px 80px}
  header{border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:18px}
  h1{font-family:"Barlow Condensed",sans-serif;font-weight:700;letter-spacing:.02em;
     text-transform:uppercase;font-size:30px;margin:0;line-height:1}
  .sub{color:var(--muted);font-size:13px;margin-top:6px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .badge{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;letter-spacing:.06em;
         font-size:11px;padding:2px 8px;border:1px solid var(--line);border-radius:2px;color:var(--muted)}
  .tabs{display:flex;gap:6px;margin:0 0 18px;flex-wrap:wrap}
  .tab{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;letter-spacing:.04em;
       font-size:15px;font-weight:600;padding:7px 14px;background:var(--panel);border:1px solid var(--line);
       color:var(--muted);border-radius:3px;cursor:pointer}
  .tab.on{color:var(--text);border-color:var(--home);background:var(--panel2)}
  .honesty{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--draw);
           border-radius:4px;padding:12px 14px;margin-bottom:16px;font-size:13px;color:var(--muted);
           display:flex;gap:20px;flex-wrap:wrap;align-items:baseline}
  .honesty b{color:var(--text);font-weight:600}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:14px 15px;margin-bottom:11px}
  .row{display:flex;justify-content:space-between;align-items:baseline;gap:10px}
  .teams{font-family:"Barlow Condensed",sans-serif;font-size:20px;font-weight:600;line-height:1.15}
  .vs{color:var(--muted);font-weight:500;padding:0 4px}
  .date{color:var(--muted);font-size:12px;white-space:nowrap}
  .form{display:flex;gap:8px;margin:7px 0 12px;font-size:12px;color:var(--muted)}
  .pill{letter-spacing:1px;font-weight:600}
  .W{color:var(--home)} .D{color:var(--draw)} .L{color:var(--loss)}
  .news{font-size:12px;color:#cdb890;background:#1e1a12;border:1px solid #4a3d22;border-radius:3px;
        padding:6px 9px;margin:-2px 0 11px;line-height:1.45}
  .news b{color:#e8d3b0;font-weight:600}
  .bar{display:flex;height:26px;border-radius:3px;overflow:hidden;margin-bottom:4px}
  .seg{display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;color:#0d1117;min-width:34px}
  .seg.h{background:var(--home)} .seg.d{background:var(--draw)} .seg.a{background:var(--away)}
  .barkey{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-bottom:12px}
  .chips{display:flex;gap:7px;flex-wrap:wrap}
  .chip{background:var(--panel2);border:1px solid var(--line);border-radius:3px;padding:4px 9px;font-size:12px;color:var(--muted)}
  .chip b{color:var(--text);font-weight:600}
  .scores{display:flex;gap:7px;margin-top:11px}
  .score{font-family:"Barlow Condensed",sans-serif;font-weight:600;font-size:14px;background:var(--panel2);
         border:1px solid var(--line);border-radius:3px;padding:3px 8px;color:var(--text)}
  .score span{color:var(--muted);font-family:Inter;font-weight:400;font-size:11px;margin-left:5px}
  .derby{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;letter-spacing:.05em;
         font-size:11px;font-weight:600;color:var(--draw);border:1px solid var(--draw);
         border-radius:2px;padding:1px 6px;vertical-align:middle;white-space:nowrap}
  .h2h{margin-top:10px;font-size:12px;color:var(--muted)}
  .h2h b{color:var(--text);font-weight:600} .h2h span{opacity:.7}
  .accrow{display:flex;align-items:center;gap:11px;background:var(--panel);border:1px solid var(--line);
          border-radius:5px;padding:10px 13px;margin-bottom:7px;cursor:pointer}
  .accrow.sel{border-color:var(--home);background:var(--panel2)}
  .accrow input{width:17px;height:17px;accent-color:var(--home);flex:none}
  .accrow .m{flex:1;font-family:"Barlow Condensed",sans-serif;font-size:16px;font-weight:600}
  .accrow .m span{font-family:Inter;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;display:block;font-weight:400}
  .accrow .n{text-align:right;font-size:12px;color:var(--muted);white-space:nowrap}
  .accrow .n b{color:var(--text);font-size:15px;font-family:"Barlow Condensed",sans-serif}
  .quick{display:flex;gap:7px;margin-bottom:14px}
  .quick button{font-family:"Barlow Condensed",sans-serif;text-transform:uppercase;letter-spacing:.04em;
    font-size:13px;font-weight:600;padding:6px 12px;background:var(--panel2);border:1px solid var(--line);
    color:var(--muted);border-radius:3px;cursor:pointer}
  .accsum{position:sticky;top:0;background:var(--panel2);border:1px solid var(--home);border-radius:5px;
    padding:13px 15px;margin-bottom:12px;display:flex;flex-wrap:wrap;gap:18px}
  .accsum div{font-size:12px;color:var(--muted)} .accsum b{display:block;color:var(--text);
    font-family:"Barlow Condensed",sans-serif;font-size:22px;font-weight:700;margin-top:2px}
  .accread{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--draw);
    border-radius:4px;padding:11px 14px;margin-bottom:14px;font-size:13px;line-height:1.5;color:var(--muted)}
  .warn{background:#241a12;border:1px solid #6b4a1f;border-left:3px solid var(--draw);
        border-radius:4px;padding:13px 15px;margin-bottom:16px;font-size:13px;line-height:1.55;color:#e8d3b0}
  .warn b{color:#fff}
  .vrow{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:12px 14px;margin-bottom:9px;
        display:flex;justify-content:space-between;align-items:center;gap:12px}
  .vpick{font-family:"Barlow Condensed",sans-serif;font-size:17px;font-weight:600}
  .vpick .lg{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;display:block;margin-top:2px}
  .vnums{text-align:right;font-size:12px;color:var(--muted);white-space:nowrap}
  .vedge{font-family:"Barlow Condensed",sans-serif;font-size:20px;font-weight:700;color:var(--home)}
  .res{margin-top:11px;padding-top:10px;border-top:1px solid var(--line);font-size:13px;color:var(--muted)}
  .res b{font-family:"Barlow Condensed",sans-serif;font-size:16px;color:var(--text)}
  .hit{color:var(--home)} .miss{color:var(--loss)}
  footer{margin-top:26px;color:var(--muted);font-size:12px;line-height:1.6;border-top:1px solid var(--line);padding-top:14px}
  @media (prefers-reduced-motion:no-preference){.card{transition:border-color .15s}.card:hover{border-color:var(--home)}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Lower-League Model</h1>
    <div class="sub" id="sub"></div>
  </header>
  <div class="tabs" id="tabs"></div>
  <div id="view"></div>
  <footer id="foot"></footer>
</div>
<script>
const DATA = /*DATA*/{};
const pct = x => (x*100).toFixed(0)+'%';
function formHTML(s){return [...s].map(c=>`<span class="pill ${c}">${c}</span>`).join('')}
function fixtureHTML(fx){
  const th=pct(fx.p_home),td=pct(fx.p_draw),ta=pct(fx.p_away);
  let res='';
  if(fx.result){
    const pick=Math.max(fx.p_home,fx.p_draw,fx.p_away);
    const pi=pick===fx.p_home?0:(pick===fx.p_draw?1:2);
    const [h,a]=fx.result.split('-').map(Number);
    const oc=h>a?0:(h===a?1:2);
    res=`<div class="res">Final <b>${fx.result}</b> · model favoured
         ${['home','draw','away'][pi]} <span class="${pi===oc?'hit':'miss'}">${pi===oc?'✓':'✗'}</span></div>`;
  }
  const corners=fx.corners_o95!=null?`<div class="chip">Over 9.5 corners <b>${pct(fx.corners_o95)}</b></div>`:'';
  const cards=fx.cards_o35!=null?`<div class="chip">Over 3.5 cards <b>${pct(fx.cards_o35)}</b></div>`:'';
  const bh=fx.btts_bh!=null?`<div class="chip">BTTS both halves <b>${pct(fx.btts_bh)}</b></div>`:'';
  const fh=fx.fh_over05!=null?`<div class="chip">1st-half goal <b>${pct(fx.fh_over05)}</b></div>`:'';
  const dby=fx.derby?`<span class="derby">${fx.derby}</span>`:'';
  let h2hHTML='';
  if(fx.h2h){const q=fx.h2h;h2hHTML=`<div class="h2h">H2H last ${q.n}: <b>${q.w}</b>-<b>${q.d}</b>-<b>${q.l}</b> <span>(${fx.home.split(' ')[0]} view)</span> · last ${q.last}</div>`;}
  return `<div class="card">
    <div class="row">
      <div class="teams">${fx.home}<span class="vs">v</span>${fx.away} ${dby}</div>
      <div class="date">${fx.date}${fx.time?` · ${fx.time}`:''}</div>
    </div>
    <div class="form">${formHTML(fx.form_home)} <span style="color:var(--line)">|</span> ${formHTML(fx.form_away)}</div>
    ${(fx.note_home||fx.note_away)?`<div class="news">${fx.note_home?`<b>${fx.home.split(' ')[0]}:</b> ${fx.note_home} `:''}${fx.note_away?`<b>${fx.away.split(' ')[0]}:</b> ${fx.note_away}`:''}</div>`:''}
    <div class="bar">
      <div class="seg h" style="width:${th}">${th}</div>
      <div class="seg d" style="width:${td}">${td}</div>
      <div class="seg a" style="width:${ta}">${ta}</div>
    </div>
    <div class="barkey"><span>Home win</span><span>Draw</span><span>Away win</span></div>
    <div class="chips">
      <div class="chip">Over 2.5 <b>${pct(fx.over25)}</b></div>
      <div class="chip">BTTS <b>${pct(fx.btts)}</b></div>
      ${bh}${fh}
      <div class="chip">Clean sheet H <b>${pct(fx.cs_home)}</b> · A <b>${pct(fx.cs_away)}</b></div>
      ${corners}${cards}
    </div>
    <div class="scores">${fx.tops.map(t=>`<div class="score">${t.score}<span>${pct(t.p)}</span></div>`).join('')}</div>
    ${h2hHTML}
    ${res}
  </div>`;
}
function leagueHTML(lg){
  let h='';
  const hp=lg.honesty;
  if(hp){
    h+=`<div class="honesty">
      <span>Backtest (last 90 days, ${hp.n} matches): model RPS <b>${hp.model_rps}</b> vs prior <b>${hp.base_rps}</b> — lower is better.</span>
      ${hp.btts_pred!=null?`<span>BTTS calibration: predicted <b>${hp.btts_pred}</b> → observed <b>${hp.btts_obs}</b>.</span>`:''}</div>`;
  }
  if(!lg.has_cards) h+=`<div class="honesty" style="border-left-color:var(--away)">Cards, corners and closing-line value fill in automatically on the live football-data feed — this view is the free results mirror.</div>`;
  h+=lg.fixtures.map(fixtureHTML).join('');
  if(!lg.fixtures.length) h+=`<div class="card">No upcoming fixtures right now — this usually just means the next round hasn't been published yet, or games are mid-week. Check back after the current round.</div>`;
  return h;
}
function valueHTML(){
  const warn=`<div class="warn"><b>Read this first.</b> These are fixtures where the model most
    disagrees with the bookmaker's price — not tips. The model has <b>not</b> been shown to beat the
    closing line, so a flagged gap is as likely to be model error as a real edge. Treat each as a
    hypothesis: note the price now, check where it closes, and only trust the signal if your
    closing-line value is positive over a real sample. No stakes advice here by design.</div>`;
  let picks=[];
  DATA.leagues.forEach(l=>l.fixtures.forEach(f=>{if(f.value)picks.push({...f,lg:l.name});}));
  picks.sort((a,b)=>b.value.edge-a.value.edge);
  if(!picks.length) return warn+`<div class="card">No odds in this view, so nothing to compare.
    The Model vs Market screen activates on the live feed (<code>python app.py</code>), which carries
    bookmaker prices. This demo runs on the free results mirror, which has none.</div>`;
  return warn+picks.map(f=>{const v=f.value;return `<div class="vrow">
    <div><div class="vpick">${v.pick}: ${f.home} v ${f.away}<span class="lg">${f.lg} · ${f.date}</span></div></div>
    <div class="vnums">model <b style="color:var(--text)">${pct(v.model)}</b> vs market ${pct(v.market)} · odds ${v.odds}
      <div class="vedge">+${pct(v.edge)}</div></div></div>`;}).join('');
}
let accaSel=new Set();
function accaCandidates(){
  let c=[];
  DATA.leagues.forEach(l=>l.fixtures.forEach(f=>{
    const p=[f.p_home,f.p_draw,f.p_away]; const i=p.indexOf(Math.max(...p));
    c.push({id:`${f.home}|${f.away}|${f.date}`,lg:l.name,home:f.home,away:f.away,date:f.date,
            pick:['Home','Draw','Away'][i],model:p[i],
            odds:f.odds3?f.odds3[i]:null,mkt:f.mkt3?f.mkt3[i]:null});
  }));
  return c.sort((a,b)=>b.model-a.model);
}
function accaSummary(){
  const legs=accaCandidates().filter(c=>accaSel.has(c.id));
  if(!legs.length) return '<div class="accread">Tick legs below (or use the quick-picks) to build a ticket. Each leg multiplies the all-must-win chance down.</div>';
  let mp=1,haveOdds=legs.every(l=>l.odds),odds=1,mkt=1;
  legs.forEach(l=>{mp*=l.model;if(haveOdds){odds*=l.odds;mkt*=l.mkt;}});
  let s=`<div class="accsum"><div>Legs<b>${legs.length}</b></div><div>Model: it all lands<b>${pct(mp)}</b></div>`;
  let read;
  if(haveOdds){
    const ev=mp*odds-1;
    s+=`<div>Combined odds<b>${odds.toFixed(2)}</b></div><div>Book-implied chance<b>${pct(mkt)}</b></div>`+
       `<div>Model edge<b class="${ev>=0?'hit':'miss'}">${(ev*100).toFixed(0)}%</b></div>`;
    read = ev>=0
      ? 'Model shows positive expectation — but it rests on probabilities not yet proven to beat the close, and the margin still compounds with every leg. Hypothesis, not a green light: log the CLV.'
      : `Negative expectation: the model reckons this ticket is priced against you by ${(-ev*100).toFixed(0)}%. That gap is the compounded bookmaker margin — it grows every time you add a leg. This is the normal result for an acca.`;
  } else {
    read='Odds appear on the live feed, which unlocks the expected-value read. From model probability alone: '+legs.length+' legs at these confidences all landing is '+pct(mp)+'.';
  }
  s+=`</div><div class="accread">${read}</div>`;
  return s;
}
function accaToggle(id){ accaSel.has(id)?accaSel.delete(id):accaSel.add(id); refreshAcca(); }
function accaTop(n){ accaSel=new Set(accaCandidates().slice(0,n).map(c=>c.id)); refreshAcca(); }
function refreshAcca(){
  document.getElementById('accsummary').innerHTML=accaSummary();
  document.querySelectorAll('.accrow').forEach(r=>r.classList.toggle('sel',accaSel.has(r.dataset.id)));
  document.querySelectorAll('.accrow input').forEach(cb=>cb.checked=accaSel.has(cb.closest('.accrow').dataset.id));
}
function accaHTML(){
  const c=accaCandidates();
  const warn=`<div class="warn"><b>How accumulators actually work.</b> Ranking by most-likely result surfaces
    short-priced favourites the book prices about the same as the model — no edge, and the vig compounds with every
    leg (a 10-fold can carry 30%+ margin). "Most likely" is not "best value". Use this to see what a ticket really
    is, not as a tipsheet.</div>`;
  const quick=`<div class="quick">Quick-pick:
    <button onclick="accaTop(3)">Top 3</button><button onclick="accaTop(5)">Top 5</button>
    <button onclick="accaTop(10)">Top 10</button><button onclick="accaSel=new Set();refreshAcca()">Clear</button></div>`;
  const rows=c.map(x=>`<label class="accrow" data-id="${x.id}">
      <input type="checkbox" onchange="accaToggle('${x.id}')">
      <div class="m">${x.pick}: ${x.home} v ${x.away}<span>${x.lg} · ${x.date}</span></div>
      <div class="n">model <b>${pct(x.model)}</b>${x.odds?` · odds ${x.odds}`:''}</div>
    </label>`).join('');
  return warn+`<div id="accsummary">${accaSummary()}</div>`+quick+rows;
}
function show(i){
  const tabs=document.querySelectorAll('.tab');
  tabs.forEach((t,j)=>t.classList.toggle('on',i===j));
  const v=document.getElementById('view');
  if(i<DATA.leagues.length) v.innerHTML=leagueHTML(DATA.leagues[i]);
  else if(i===DATA.leagues.length) v.innerHTML=valueHTML();
  else { v.innerHTML=accaHTML(); refreshAcca(); }
}
document.getElementById('sub').innerHTML=
  `<span>Generated ${DATA.generated}</span><span class="badge">${DATA.mode}</span>`;
document.getElementById('tabs').innerHTML=
  DATA.leagues.map((l,i)=>`<button class="tab" onclick="show(${i})">${l.name}</button>`).join('')
  + `<button class="tab" onclick="show(${DATA.leagues.length})">⚑ Model vs Market</button>`
  + `<button class="tab" onclick="show(${DATA.leagues.length+1})">Σ Acca Lab</button>`;
document.getElementById('foot').innerHTML=
  'Probabilities are calibrated estimates, not certainties — in these tiers the goals signal is thin, '+
  'so treat the model as a base rate and lineups, injuries and referees as the real edge. '+
  'Score yourself against the closing line (CLV), not against your own baseline.';
if(DATA.leagues.length) show(0);
</script>
</body>
</html>
"""

if __name__=="__main__":
    main()
