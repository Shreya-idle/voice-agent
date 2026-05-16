import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Mic, MicOff, Send, Volume2, VolumeX, Bot, User, Loader2, PhoneOff, Zap } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { signInAnonymously, onAuthStateChanged } from 'firebase/auth';
import { auth } from './firebase';

// LiveKit imports
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useRoomContext,
  useTracks,
  useLocalParticipant,
  BarVisualizer,
} from '@livekit/components-react';
import { Track, RoomEvent } from 'livekit-client';

const API_URL = 'http://localhost:8000';
const LIVEKIT_URL = import.meta.env.VITE_LIVEKIT_URL;

function App() {
  const [uid, setUid] = useState(null);
  const [livekitToken, setLivekitToken] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  
  useEffect(() => {
    signInAnonymously(auth).catch(error => {
      console.error("Error signing in anonymously:", error);
    });

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        setUid(user.uid);
      } else {
        setUid(null);
      }
    });

    return () => unsubscribe();
  }, []);

  const handleConnect = async () => {
    if (!uid) return;
    try {
      const roomName = `room-${uid.substring(0, 8)}`;
      const response = await axios.get(`${API_URL}/token`, {
        params: { room: roomName, identity: uid }
      });
      setLivekitToken(response.data.token);
      setIsConnected(true);
    } catch (error) {
      console.error("Error fetching LiveKit token:", error);
    }
  };

  const handleDisconnect = () => {
    setIsConnected(false);
    setLivekitToken(null);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4 sm:p-8">
      {isConnected && livekitToken ? (
        <LiveKitRoom
          serverUrl={LIVEKIT_URL}
          token={livekitToken}
          connect={true}
          audio={true}
          video={false}
          onDisconnected={handleDisconnect}
          className="w-full max-w-4xl h-[85vh] flex flex-col glass-card overflow-hidden"
        >
          <VoiceAgentUI uid={uid} onDisconnect={handleDisconnect} />
          <RoomAudioRenderer />
        </LiveKitRoom>
      ) : (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="w-full max-w-md p-8 glass-card text-center"
        >
          <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-sky-400 to-indigo-500 flex items-center justify-center mx-auto mb-6 shadow-lg shadow-indigo-500/20">
            <Bot size={40} color="white" />
          </div>
          <h1 className="text-3xl font-bold mb-2">Voice Agent</h1>
          <p className="text-slate-400 mb-8">Connect to start a real-time conversation with your personal AI assistant.</p>
          <button 
            onClick={handleConnect}
            disabled={!uid}
            className="w-full btn-primary py-4 rounded-2xl font-bold flex items-center justify-center gap-2 group"
          >
            {!uid ? <Loader2 className="animate-spin" /> : <Zap size={20} className="group-hover:animate-pulse" />}
            {uid ? "Start Conversation" : "Initializing..."}
          </button>
        </motion.div>
      )}
    </div>
  );
}

function VoiceAgentUI({ uid, onDisconnect }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Connected! I am listening. How can I help you?' }
  ]);
  const [input, setInput] = useState('');
  const [credits, setCredits] = useState('...');
  const [analytics, setAnalytics] = useState(null);
  const scrollRef = useRef(null);
  
  const room = useRoomContext();
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant();
  
  // Track remote audio for visualizer
  const tracks = useTracks([Track.Source.Microphone]);
  const agentTrack = tracks.find(t => t.participant.identity !== uid);

  useEffect(() => {
    fetchAnalytics();
    
    // Listen for data messages (transcripts) if the agent sends them
    const handleData = (payload, participant) => {
      const decoder = new TextDecoder();
      const text = decoder.decode(payload);
      try {
        const data = JSON.parse(text);
        if (data.type === 'transcript') {
          setMessages(prev => [...prev, { role: data.role, content: data.content }]);
        }
      } catch (e) {
        // Fallback if not JSON
        console.log("Received raw message:", text);
      }
    };

    room.on(RoomEvent.DataReceived, handleData);
    return () => {
      room.off(RoomEvent.DataReceived, handleData);
    };
  }, [room]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const fetchAnalytics = async () => {
    try {
      const response = await axios.get(`${API_URL}/analytics`);
      setAnalytics(response.data);
    } catch (error) {
      console.error("Error fetching analytics:", error);
    }
  };

  const toggleMic = async () => {
    if (localParticipant) {
      await localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled);
    }
  };

  return (
    <>
      <div className="p-6 border-b border-[rgba(255,255,255,0.1)] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-sky-400 to-indigo-500 flex items-center justify-center">
            <Bot size={24} color="white" />
          </div>
          <div>
            <h2 className="text-xl font-bold">Voice Agent</h2>
            <p className="text-xs text-emerald-400 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Real-time | {room.state}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-1 px-3 py-1 rounded-full bg-white/5 border border-white/10">
            <span className="text-xs font-medium text-slate-400">RAG Enabled</span>
          </div>
          <button 
            onClick={onDisconnect}
            className="p-2 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 transition-all"
          >
            <PhoneOff size={18} />
          </button>
        </div>
      </div>

      {analytics && (
        <div className="bg-white/5 border-b border-white/10 px-6 py-2 flex items-center justify-around">
          <div className="text-center">
            <p className="text-[10px] text-slate-400 uppercase">Total Questions</p>
            <p className="text-sm font-bold text-white">{analytics.total_questions}</p>
          </div>
          <div className="w-[1px] h-6 bg-white/10"></div>
          <div className="text-center">
            <p className="text-[10px] text-slate-400 uppercase">Answered</p>
            <p className="text-sm font-bold text-emerald-400">{analytics.answered}</p>
          </div>
          <div className="w-[1px] h-6 bg-white/10"></div>
          <div className="text-center">
            <p className="text-[10px] text-slate-400 uppercase">Success Rate</p>
            <p className="text-sm font-bold text-sky-400">{Math.round(analytics.success_rate)}%</p>
          </div>
        </div>
      )}

      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth"
      >
        {messages.map((msg, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex gap-3 max-w-[80%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'user' ? 'bg-indigo-500' : 'bg-white/10'}`}>
                {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div className={`p-4 rounded-2xl ${msg.role === 'user' ? 'bg-indigo-600/20 border border-indigo-500/30' : 'bg-white/5 border border-white/10'}`}>
                <p className="text-sm leading-relaxed">{msg.content}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="p-6 bg-white/5 border-t border-white/10">
        <div className="flex flex-col items-center gap-6">
          {/* Visualizer for Agent's Voice */}
          <div className="w-full h-16 flex items-center justify-center">
            {agentTrack ? (
              <BarVisualizer 
                trackRef={agentTrack}
                barCount={20}
                gap={4}
                className="h-full w-full max-w-sm fill-sky-400"
              />
            ) : (
              <div className="flex gap-1">
                {Array.from({ length: 15 }).map((_, i) => (
                  <div key={i} className="w-1 h-2 bg-white/10 rounded-full" />
                ))}
              </div>
            )}
          </div>

          <div className="flex items-center gap-4 w-full justify-center">
            <button 
              onClick={toggleMic}
              className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                isMicrophoneEnabled 
                ? 'bg-indigo-600 shadow-lg shadow-indigo-500/40' 
                : 'bg-white/10 border border-white/10'
              }`}
            >
              {isMicrophoneEnabled ? <Mic size={28} /> : <MicOff size={28} className="text-slate-400" />}
            </button>
            
            <div className="flex flex-col">
              <p className="text-sm font-medium text-white">
                {isMicrophoneEnabled ? "Listening..." : "Microphone Muted"}
              </p>
              <p className="text-xs text-slate-400">
                {isMicrophoneEnabled ? "Speak naturally to chat" : "Tap to unmute"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
