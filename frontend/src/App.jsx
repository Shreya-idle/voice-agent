import { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { auth } from './firebase';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useRoomContext,
  useTracks,
  useLocalParticipant,
} from '@livekit/components-react';
import { Track, RoomEvent } from 'livekit-client';
import './App.css';

const API_URL      = import.meta.env.VITE_BACKEND_URL  
const LIVEKIT_URL  = import.meta.env.VITE_LIVEKIT_URL  

const CLONE_STAGES = [
  { label: 'Stage 1', kicker: 'Voice waveform / microphone', title: 'Analyzing your voice…', detail: 'Capturing vocal tone and cadence', type: 'voice' },
  { label: 'Stage 2', kicker: 'AI thinking / processing', title: 'Understanding voice characteristics…', detail: 'Mapping pitch, pace, and expression', type: 'brain' },
  { label: 'Stage 3', kicker: 'Voice cloning / neural processing', title: 'Generating your cloned voice…', detail: 'Training a unique neural voice model', type: 'neural' },
  { label: 'Stage 4', kicker: 'Enhancing / polishing voice', title: 'Polishing the audio…', detail: 'Balancing clarity and natural resonance', type: 'polish' },
  { label: 'Stage 5', kicker: 'Audio waveform generation', title: 'Adding natural expression…', detail: 'Finishing the final listening experience', type: 'finish' },
];

function StageVisual({ type }) {
  if (type === 'voice' || type === 'polish') {
    return <div className="clone-wave" aria-hidden="true">{Array.from({ length: 17 }, (_, index) => <i key={index} />)}</div>;
  }
  if (type === 'brain') return <div className="clone-brain" aria-hidden="true"><span>✦</span><b>◌</b><em>✧</em></div>;
  if (type === 'neural') return <div className="clone-neural" aria-hidden="true"><div className="dna">⌁</div><span /><span /><span /><span /><span /></div>;
  return <div className="clone-finish" aria-hidden="true"><span /><span /><span /><b>✓</b></div>;
}

function CloneFlow({ onComplete }) {
  const [activeStage, setActiveStage] = useState(0);
  const stage = CLONE_STAGES[activeStage];

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (activeStage === CLONE_STAGES.length - 1) onComplete();
      else setActiveStage(current => current + 1);
    }, 2600);
    return () => window.clearTimeout(timer);
  }, [activeStage, onComplete]);

  if (window.__cloneFlowLegacy) return (
    <main className={`clone-flow clone-flow--${stage.type}`}>
      <div className="clone-orbit clone-orbit--one" />
      <div className="clone-orbit clone-orbit--two" />
      <header className="clone-header">
        <div className="clone-brand"><span>◈</span> CLONEVOX</div>
        <button className="clone-skip" type="button" onClick={onComplete}>Skip intro</button>
      </header>
      <section className="clone-content" aria-live="polite">
        <div className="clone-stage-label">{stage.label}<span> / 5</span></div>
        <AnimatePresence mode="wait">
          <motion.div
            className="clone-stage"
            key={stage.type}
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.98 }}
            transition={{ duration: 0.42, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="clone-visual"><StageVisual type={stage.type} /></div>
            <p className="clone-kicker">{stage.kicker}</p>
            <h1>{stage.title}</h1>
            <p className="clone-detail">{stage.detail}</p>
          </motion.div>
        </AnimatePresence>
        <div className="clone-progress-wrap">
          <div className="clone-progress" key={activeStage}><span /></div>
          <div className="clone-progress-caption"><span>Generation in progress</span><span>{Math.round(((activeStage + 1) / 5) * 100)}%</span></div>
        </div>
      </section>
      <nav className="clone-steps" aria-label="Voice generation progress">
        {CLONE_STAGES.map((item, index) => <div className={`clone-step ${index === activeStage ? 'is-active' : ''} ${index < activeStage ? 'is-complete' : ''}`} key={item.type}><span>{index < activeStage ? '✓' : `0${index + 1}`}</span><i /></div>)}
      </nav>
    </main>
  );
  return <FlowOverview stage={stage} activeStage={activeStage} setActiveStage={setActiveStage} onComplete={onComplete} />;
}

function FlowOverview({ stage, activeStage, setActiveStage, onComplete }) {
  return <main className="clone-flow flow-overview" aria-label="CloneVox voice generation flow">
    <header className="flow-header"><div><div className="flow-brand">CLONEVOX</div><h1>GENERATION FLOW</h1></div><button className="flow-skip" type="button" onClick={onComplete}>Skip animation</button></header>
    <section className="flow-canvas" aria-live="polite">
      <svg className="flow-lines" viewBox="0 0 1200 620" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="flowLine" x1="0" x2="1"><stop stopColor="#4ee1df"/><stop offset=".52" stopColor="#718eff"/><stop offset="1" stopColor="#9d6bff"/></linearGradient></defs><path d="M210 290H318M472 290H555M696 290H798M951 290H1080M923 244V104H1080M210 338V502H536M1080 502H760" /></svg>
      <div className="flow-side flow-side--left">{CLONE_STAGES.slice(0, 2).map((item, index) => <FlowCard item={item} index={index} activeStage={activeStage} key={item.type} />)}</div>
      <div className="flow-phone"><div className="phone-frame"><div className="phone-island" /><div className="phone-status">10:09 AM <span>|||</span></div><AnimatePresence mode="wait"><motion.div className="phone-content" key={stage.type} initial={{ opacity: 0, scale: .94 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 1.04 }} transition={{ duration: .36 }}><div className="phone-art"><StageVisual type={stage.type} /></div><div className="phone-stage">{stage.label}</div><p>{stage.kicker}</p><h2>{stage.title}</h2><div className="phone-progress"><i /></div><small>[2-3 sec]</small></motion.div></AnimatePresence><div className="phone-dock"><div><span>Generation progress</span><b>{Math.round(((activeStage + 1) / 5) * 100)}%</b></div><div className="dock-progress"><i style={{ width: `${((activeStage + 1) / 5) * 100}%` }} /></div><em>Estimated time remaining: ~{(5 - activeStage) * 3} sec</em></div></div></div>
      <div className="flow-side flow-side--right">{CLONE_STAGES.slice(3).map((item, offset) => <FlowCard item={item} index={offset + 3} activeStage={activeStage} key={item.type} />)}</div>
    </section>
    <nav className="flow-mobile-steps" aria-label="Generation progress">{CLONE_STAGES.map((item, index) => <button type="button" onClick={() => setActiveStage(index)} className={index === activeStage ? 'is-active' : index < activeStage ? 'is-done' : ''} key={item.type}>{index + 1}</button>)}</nav>
  </main>;
}

function FlowCard({ item, index, activeStage }) {
  const isActive = index === activeStage;
  return <motion.article className={`flow-card ${isActive ? 'is-active' : ''} ${index < activeStage ? 'is-complete' : ''}`} animate={{ opacity: isActive ? 1 : .58, y: isActive ? -4 : 0 }} transition={{ duration: .3 }}><div className="flow-card-label">Stage {index + 1}</div><div className="flow-card-art"><StageVisual type={item.type} /><b>{index + 1}</b></div><p>{item.kicker}</p><h2>{item.title}</h2><div className="flow-card-progress"><i style={{ width: isActive || index < activeStage ? '100%' : '0%' }} /></div><small>[2-3 sec]</small></motion.article>;
}

const IconBot = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="10" rx="2"/><path d="M12 11V5"/><circle cx="12" cy="4" r="1"/>
    <path d="M8 15h.01M16 15h.01"/>
  </svg>
);
const IconUser = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
  </svg>
);
const IconMic = ({ off }) => off
  ? <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="2" y1="2" x2="22" y2="22"/><path d="M18.89 13.23A7.12 7.12 0 0 0 19 12v-2"/><path d="M5 10v2a7 7 0 0 0 12 4.93"/><path d="M9 9v3a3 3 0 0 0 5.12 2.12"/><rect x="9" y="2" width="6" height="10" rx="3"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
  : <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>;
const IconSend = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);
const IconZap = () => (
  <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" stroke="none">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
  </svg>
);
const IconLoader = () => (
  <svg className="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
  </svg>
);
const IconPhone = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.42 19.42 0 0 1 4.43 9.68 19.79 19.79 0 0 1 1.36 1a2 2 0 0 1 2-2.18h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11z"/>
    <line x1="1" y1="1" x2="23" y2="23"/>
  </svg>
);

function WaveSticks({ count = 9, active, side }) {
  const [heights, setHeights] = useState(() => Array(count).fill(4));
  useEffect(() => {
    if (!active) {
      const resetId = setTimeout(() => setHeights(Array(count).fill(4)), 0);
      return () => clearTimeout(resetId);
    }
    const id = setInterval(() => {
      setHeights(prev => prev.map(() => Math.floor(Math.random() * 20) + 3));
    }, 120);
    return () => clearInterval(id);
  }, [active, count]);

  const ordered = side === 'right' ? [...heights].reverse() : heights;
  return (
    <div className="waveform">
      {ordered.map((h, i) => (
        <div key={i} className={`wave-stick ${active ? 'active' : ''}`} style={{ height: h }} />
      ))}
    </div>
  );
}

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function App() {
  const [uid, setUid]               = useState(null);
  const [firebaseUser, setFirebaseUser] = useState(null);
  const [livekitToken, setToken]    = useState(null);
  const [isConnected, setConnected] = useState(false);
  const [introComplete, setIntroComplete] = useState(false);

  useEffect(() => {
    signInAnonymously(auth).catch(console.error);
    const unsub = onAuthStateChanged(auth, u => {
      setFirebaseUser(u);
      setUid(u?.uid ?? null);
    });
    return unsub;
  }, []);

  const handleDisconnect = useCallback(() => {
    setConnected(false);
    setToken(null);
  }, []);

  const fetchAndSetToken = useCallback(async () => {
    if (!uid || !firebaseUser) return false;
    try {
      const room = `room-${uid.substring(0, 8)}`;
      const fbToken = await firebaseUser.getIdToken(true); // Force refresh Firebase token
      if (!fbToken) {
        throw new Error('Firebase ID token is missing');
      }
      const { data } = await axios.get(`${API_URL}/token`, {
        params: { room, identity: uid },
        headers: {
          'Cache-Control': 'no-cache',
          'Authorization': `Bearer ${fbToken}`
        },
      });
      setToken(data.token);
      console.log("Token fetched/refreshed");
      return true;
    } catch (e) {
      console.error('Token fetch error:', e.response?.data ?? e);
      handleDisconnect();
      return false;
    }
  }, [uid, firebaseUser, handleDisconnect]);

  useEffect(() => {
    if (!livekitToken) return;

    const decodedToken = JSON.parse(atob(livekitToken.split('.')[1]));
    const expirationTime = decodedToken.exp * 1000;
    const timeout = expirationTime - Date.now() - 60000; // Refresh 1 minute before expiry

    if (timeout <= 0) {
      const refreshId = setTimeout(fetchAndSetToken, 0);
      return () => clearTimeout(refreshId);
    }

    const timer = setTimeout(fetchAndSetToken, timeout);
    return () => clearTimeout(timer);
  }, [livekitToken, fetchAndSetToken]);

  const handleConnect = async () => {
    const ok = await fetchAndSetToken();
    setConnected(ok);
  };

  if (!introComplete) return <CloneFlow onComplete={() => setIntroComplete(true)} />;

  if (isConnected && livekitToken) {
    return (
      <LiveKitRoom
        serverUrl={LIVEKIT_URL}
        token={livekitToken}
        connect={true}
        audio={true}
        video={false}
        onDisconnected={handleDisconnect}
        style={{ width: '100vw', height: '100vh', display: 'contents' }}
      >
        <ChatShell uid={uid} onDisconnect={handleDisconnect} />
        <RoomAudioRenderer />
      </LiveKitRoom>
    );
  }

  return (
    <div className="connect-wrap">
      <div className="connect-avatar">
        <IconBot />
      </div>
      <div>
        <div className="connect-title">Voice Agent</div>
      </div>
      <div className="connect-sub">
        Connect to start a real-time conversation with your personal AI assistant.
      </div>
      <button className="btn-connect" onClick={handleConnect} disabled={!uid}>
        {!uid ? <IconLoader /> : <IconZap />}
        {uid ? 'Start Conversation' : 'Initializing...'}
      </button>
    </div>
  );
}

function ChatShell({ uid, onDisconnect }) {
  const [conversationId]      = useState(() => crypto.randomUUID());
  const [messages, setMessages]   = useState([
    { role: 'agent', content: 'Connected! I am listening. How can I help you?', time: nowTime() },
  ]);
  const [input, setInput]         = useState('');
  const [sending, setSending]     = useState(false);
  const [credits, setCredits]     = useState('...');
  const [analytics, setAnalytics] = useState(null);
  const scrollRef                 = useRef(null);

  const room                                    = useRoomContext();
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant();
  const tracks  = useTracks([Track.Source.Microphone]);
  const agentOn = tracks.some(t => t.participant.identity !== uid);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) setTimeout(() => { el.scrollTop = el.scrollHeight; }, 40);
  }, [messages]);

  const fetchCredits = useCallback(async () => {
    if (!uid) return;
    try { 
      const token = await auth.currentUser?.getIdToken();
      const { data } = await axios.get(`${API_URL}/user/${uid}/credits`, {
        headers: { 'Authorization': `Bearer ${token}` }
      }); 
      setCredits(data.credits); 
    }
    catch (error) { console.error('Credits fetch error:', error); }
  }, [uid]);
  const fetchAnalytics = useCallback(async () => {
    try { 
      
      const token = await auth.currentUser?.getIdToken();
      const { data } = await axios.get(`${API_URL}/analytics`, {
        headers: { 'Authorization': `Bearer ${token}` }

      }); 
      setAnalytics(data); 
    }
    catch (error) { console.error('Analytics fetch error:', error); }
  }, []);

  useEffect(() => {
    const loadId = setTimeout(() => {
      fetchCredits();
      fetchAnalytics();
    }, 0);
    const onData = (payload) => {
      try {
        const data = JSON.parse(new TextDecoder().decode(payload));
        if (data.type === 'transcript')
          setMessages(p => [...p, { role: data.role, content: data.content, time: nowTime() }]);
      } catch (error) { console.error('Transcript message error:', error); }
    };
    room.on(RoomEvent.DataReceived, onData);
    return () => {
      clearTimeout(loadId);
      room.off(RoomEvent.DataReceived, onData);
    };
  }, [room, fetchCredits, fetchAnalytics]);

  const toggleMic = async () => {
    if (localParticipant) await localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled);
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || sending) return;
    const msg = input.trim();
    setInput('');
    setMessages(p => [...p, { role: 'user', content: msg, time: nowTime() }]);
    setSending(true);
    try {
      const token = await auth.currentUser?.getIdToken();
      const { data } = await axios.post(`${API_URL}/chat`, 
        { message: msg, uid, conversationId },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      const { response: answer, remaining_credits } = data;
      setMessages(p => [...p, { role: 'agent', content: answer, time: nowTime() }]);
      if (remaining_credits !== undefined) setCredits(remaining_credits);
      fetchAnalytics();
    } catch (error) {
      console.error('Chat request error:', error);
      setMessages(p => [...p, { role: 'agent', content: 'Error connecting to backend.', time: nowTime() }]);
    } finally { setSending(false); }
  };

  const outOfCredits = credits === 0 || credits === '0';

  return (
    <div className="shell">
      <header className="header">
        <div className="header-left">
          <div className="avatar"><IconBot /></div>
          <div>
            <div className="header-title">Voice Agent</div>
            <div className="header-status">
              <span className="dot" />
              Real-time . {room.state}
            </div>
          </div>
        </div>
        <div className="header-right">
          <div className="pill"><IconZap /> {credits} credits</div>
          <div className="pill">RAG on</div>
          <button className="btn-disconnect" onClick={onDisconnect}>
            <IconPhone /> End
          </button>
        </div>
      </header>

      {analytics && (
        <div className="stats">
          <div className="stat-cell">
            <div className="stat-label">Questions</div>
            <div className="stat-value">{analytics.total_questions}</div>
          </div>
          <div className="stat-cell">
            <div className="stat-label">Answered</div>
            <div className="stat-value">{analytics.answered}</div>
          </div>
          <div className="stat-cell">
            <div className="stat-label">Success</div>
            <div className="stat-value stat-green">{Math.round(analytics.success_rate)}%</div>
          </div>
        </div>
      )}

      <div className="messages" ref={scrollRef}>
        <div className="date-sep">Today</div>
        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.18 }}
              className={`msg-row ${m.role === 'user' ? 'user' : 'agent'}`}
            >
              <div className="msg-inner">
                <div className="msg-avatar">
                  {m.role === 'user' ? <IconUser /> : <IconBot />}
                </div>
                <div className="msg-body">
                  <div className={`bubble ${m.role === 'user' ? 'user' : 'agent'}`}>
                    {m.content}
                  </div>
                  <div className="msg-time">{m.time}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <div className="viz-bar">
        <WaveSticks count={9} active={isMicrophoneEnabled} side="left" />
        <button className={`viz-btn ${isMicrophoneEnabled ? 'mic-on' : ''}`} onClick={toggleMic}>
          <IconMic off={!isMicrophoneEnabled} />
        </button>
        <WaveSticks count={9} active={agentOn} side="right" />
      </div>

      <form className="input-row" onSubmit={handleSend}>
        <input
          className="chat-input"
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={outOfCredits ? 'Out of credits (10/10 used).' : 'Type a message or speak...'}
          disabled={sending || outOfCredits}
        />
        <button className="send-btn" type="submit" disabled={sending || !input.trim() || outOfCredits}>
          {sending ? <IconLoader /> : <IconSend />}
        </button>
      </form>
    </div>
  );
}
