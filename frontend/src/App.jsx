import React, { useState, useEffect, useRef } from 'react';
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

const API_URL      = import.meta.env.VITE_BACKEND_URL  || 'https://voiceagent-backend-production-c3fd.up.railway.app';
const LIVEKIT_URL  = import.meta.env.VITE_LIVEKIT_URL  || 'wss://voice-agent-lxdrwst6.livekit.cloud';

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
    if (!active) { setHeights(Array(count).fill(4)); return; }
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
  const [livekitToken, setToken]    = useState(null);
  const [isConnected, setConnected] = useState(false);

  useEffect(() => {
    signInAnonymously(auth).catch(console.error);
    const unsub = onAuthStateChanged(auth, u => setUid(u?.uid ?? null));
    return unsub;
  }, []);

  const handleConnect = async () => {
    if (!uid) return;
    try {
      const room = `room-${uid.substring(0, 8)}`;
      const token = await auth.currentUser?.getIdToken();
      const { data } = await axios.get(`${API_URL}/token`, {
        params: { room, identity: uid },
        headers: { 
          'Cache-Control': 'no-cache',
          'Authorization': `Bearer ${token}`
        },
      });
      setToken(data.token);
      setConnected(true);
    } catch (e) { console.error('Token error:', e); }
  };

  const handleDisconnect = () => { setConnected(false); setToken(null); };

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

  useEffect(() => {
    fetchCredits();
    fetchAnalytics();
    const onData = (payload) => {
      try {
        const data = JSON.parse(new TextDecoder().decode(payload));
        if (data.type === 'transcript')
          setMessages(p => [...p, { role: data.role, content: data.content, time: nowTime() }]);
      } catch {}
    };
    room.on(RoomEvent.DataReceived, onData);
    return () => room.off(RoomEvent.DataReceived, onData);
  }, [room]);

  const fetchCredits = async () => {
    if (!uid) return;
    try { 
      const token = await auth.currentUser?.getIdToken();
      const { data } = await axios.get(`${API_URL}/user/${uid}/credits`, {
        headers: { 'Authorization': `Bearer ${token}` }
      }); 
      setCredits(data.credits); 
    }
    catch {}
  };
  const fetchAnalytics = async () => {
    try { 
      
      const token = await auth.currentUser?.getIdToken();
      const { data } = await axios.get(`${API_URL}/analytics`, {
        headers: { 'Authorization': `Bearer ${token}` }

      }); 
      setAnalytics(data); 
    }
    catch {}
  };

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
        { message: msg, uid },
        { headers: { 'Authorization': `Bearer ${token}` } }
      );
      const { response: answer, audio_url, remaining_credits } = data;
      setMessages(p => [...p, { role: 'agent', content: answer, time: nowTime() }]);
      if (remaining_credits !== undefined) setCredits(remaining_credits);
      if (audio_url) {
        const a = new Audio(`${API_URL.replace(/\/$/, '')}${audio_url}`);
        a.play().catch(console.error);
      }
      fetchAnalytics();
    } catch {
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

