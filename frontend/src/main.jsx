import React, {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const initial = {people:0,attentivePeople:0,averageAttentionSeconds:0,sessionSeconds:0,fps:0,overall:'no_people',tracks:[],timeline:[],calibration:{active:false,calibrated:false,progress:0},status:'starting',source:'Camera 0',modelAvailable:true,modelMessage:''};
const labels = {all:'全員集中',partial:'一部集中',none:'非集中',no_people:'待機中'};
const formatTime = n => `${Math.floor(n/60).toString().padStart(2,'0')}:${Math.floor(n%60).toString().padStart(2,'0')}`;

function Card({icon,label,value,tone='blue'}) { return <div className="card"><div className={`icon ${tone}`}>{icon}</div><div><span>{label}</span><strong>{value}</strong></div></div> }

function App(){
  const [data,setData]=useState(initial); const [videoPath,setVideoPath]=useState(''); const [panel,setPanel]=useState(false);
  useEffect(()=>{let retry; const connect=()=>{const ws=new WebSocket(`ws://${location.host}/ws/metrics`); ws.onmessage=e=>setData(JSON.parse(e.data)); ws.onclose=()=>retry=setTimeout(connect,1000)}; connect(); return()=>clearTimeout(retry)},[]);
  const post=(path,body)=>fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  const timeline=useMemo(()=>{const values=[...Array(Math.max(0,180-data.timeline.length)).fill({state:'empty'}),...data.timeline]; return values},[data.timeline]);
  return <main>
    <header><div><div className="eyebrow">PRIVACY-FIRST VISION</div><h1>共同視聴時の注意力分析</h1><p>顔匿名化 / 視線方向推定 / 集中視聴時間の集計</p></div><button className="settings" onClick={()=>setPanel(!panel)}>入力設定</button></header>
    <section className="layout">
      <div className="viewer">
        <img src="/stream" alt="匿名化されたリアルタイム映像"/>
        <div className="viewerTop"><span className={data.status==='running'?'live':'warn'}>{data.status==='running'?'● LIVE':'入力待機中'}</span><span>{data.source}</span><span>{data.fps} FPS</span></div>
        {!data.modelAvailable&&<div className="error">推論モデルを利用できません<br/><small>{data.modelMessage}</small></div>}
        {data.calibration.active&&<div className="calibration"><div className="target">＋</div><h2>画面中央を見てください</h2><div className="count">{Math.max(1,Math.ceil(3-data.calibration.progress*3))}</div><div className="progress"><i style={{width:`${data.calibration.progress*100}%`}}/></div></div>}
        <div className="privacy">映像は端末内で処理され、元映像は保存されません。</div>
      </div>
      <div className="dashboard">
        <div className="cards">
          <Card icon="◉" label="現在の状態" value={labels[data.overall]} />
          <Card icon="人" label="検出人数" value={`${data.people} 人`} tone="purple"/>
          <Card icon="◎" label="現在の集中視聴人数" value={`${data.attentivePeople} 人`} />
          <Card icon="▥" label="平均集中視聴時間" value={`${data.averageAttentionSeconds} 秒`} tone="purple"/>
          <Card icon="◷" label="今回の視聴時間" value={formatTime(data.sessionSeconds)} />
          <Card icon="✓" label="注意状態判定" value={labels[data.overall]} tone="green"/>
        </div>
        <div className="timeline"><div className="sectionTitle"><h2>注意時間のタイムライン</h2><div><b/>集中 <i/>非集中</div></div><div className="bars">{timeline.map((x,i)=><span key={i} className={x.state}/>)}</div><div className="ticks"><span>180秒前</span><span>120秒前</span><span>60秒前</span><span>現在</span></div></div>
        <div className="people"><h2>視聴者ステータス</h2>{data.tracks.length?<div className="peopleGrid">{data.tracks.map(t=><div className="person" key={t.id}><b>ID {t.id}</b><strong className={t.attentive?'on':''}>{t.attentive?'注視中':'非注視'}</strong><span>頭部方向 {t.yaw}° / {t.pitch}°</span><span>集中時間 {formatTime(t.attentionSeconds)}</span></div>)}</div>:<div className="empty">視聴者を検出すると、ここに匿名の統計情報が表示されます。</div>}</div>
      </div>
    </section>
    <section className="flow"><div><b>01</b><strong>顔検出</strong><span>端末内で検出</span></div><em>→</em><div><b>02</b><strong>匿名化処理</strong><span>顔全体をモザイク</span></div><em>→</em><div><b>03</b><strong>頭部・視線推定</strong><span>画面方向を推定</span></div><em>→</em><div><b>04</b><strong>集中時間集計</strong><span>匿名データのみ保持</span></div></section>
    {panel&&<div className="modal"><div><button className="close" onClick={()=>setPanel(false)}>×</button><h2>入力・セッション設定</h2><label>ローカル動画ファイルのパス</label><input value={videoPath} onChange={e=>setVideoPath(e.target.value)} placeholder="C:\\videos\\demo.mp4"/><button onClick={()=>post('/api/source',{kind:'video',value:videoPath})}>動画を開く</button><button className="secondary" onClick={()=>post('/api/source',{kind:'camera',value:0})}>カメラ 0 に戻す</button><hr/><button onClick={()=>{post('/api/calibration/start');setPanel(false)}}>3秒キャリブレーション</button><button className="secondary" onClick={()=>post('/api/calibration/skip')}>キャリブレーションを省略</button><button className="danger" onClick={()=>post('/api/session/reset')}>セッションをリセット</button></div></div>}
  </main>
}
createRoot(document.getElementById('root')).render(<App/>);

