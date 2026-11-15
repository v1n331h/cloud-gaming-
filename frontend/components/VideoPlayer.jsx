/**
 * ============================================
 * SuperStream AI — VideoPlayer Component
 * ============================================
 * Renders the side-by-side video comparison:
 * - Left: Low-Resolution input feed
 * - Right: AI-upscaled High-Resolution output
 *
 * Uses HTML5 Canvas for direct pixel rendering
 * (lower latency than <img> tag swapping).
 *
 * Connects to the Client ML Engine WebSocket
 * and renders frames as they arrive.
 * ============================================
 */

import { useEffect, useRef, useState, useCallback } from "react";

// ── WebSocket Configuration ──
const WS_URL = "ws://localhost:8766";
const RECONNECT_DELAY = 2000; // ms
const MAX_RECONNECT_DELAY = 15000; // ms

export default function VideoPlayer({ onMetricsUpdate, onConnectionChange }) {
  const lrCanvasRef = useRef(null);
  const hrCanvasRef = useRef(null);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectDelayRef = useRef(RECONNECT_DELAY);
  const [isConnected, setIsConnected] = useState(false);
  const [lrResolution, setLrResolution] = useState("—");
  const [hrResolution, setHrResolution] = useState("—");

  /**
   * Draw a base64 JPEG image onto a canvas element.
   * Uses an off-screen Image object for async decoding.
   */
  const drawFrame = useCallback((canvasRef, base64Data, setResolution) => {
    if (!canvasRef.current || !base64Data) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();

    img.onload = () => {
      // ── Resize canvas to match image dimensions ──
      if (canvas.width !== img.width || canvas.height !== img.height) {
        canvas.width = img.width;
        canvas.height = img.height;
        setResolution(`${img.width}×${img.height}`);
      }

      // ── Draw frame ──
      ctx.drawImage(img, 0, 0);
    };

    img.onerror = () => {
      console.warn("Failed to decode frame image");
    };

    img.src = `data:image/jpeg;base64,${base64Data}`;
  }, []);

  /**
   * Establish WebSocket connection to the Client ML Engine.
   * Handles auto-reconnection with exponential backoff.
   */
  const connect = useCallback(() => {
    // ── Prevent duplicate connections ──
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("✅ Connected to Inference Server");
        setIsConnected(true);
        onConnectionChange?.(true);
        reconnectDelayRef.current = RECONNECT_DELAY; // Reset backoff
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "frame") {
            // ── Render LR and HR frames ──
            drawFrame(lrCanvasRef, data.lr_frame, setLrResolution);
            drawFrame(hrCanvasRef, data.hr_frame, setHrResolution);

            // ── Forward metrics to parent ──
            if (data.metrics) {
              onMetricsUpdate?.(data.metrics);
            }
          }
        } catch (err) {
          console.error("Frame parse error:", err);
        }
      };

      ws.onclose = (event) => {
        console.log(`🔌 Disconnected (code: ${event.code})`);
        setIsConnected(false);
        onConnectionChange?.(false);
        scheduleReconnect();
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
      };
    } catch (err) {
      console.error("Connection failed:", err);
      scheduleReconnect();
    }
  }, [drawFrame, onMetricsUpdate, onConnectionChange]);

  /**
   * Schedule a reconnection attempt with exponential backoff.
   */
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
    }

    const delay = reconnectDelayRef.current;
    console.log(`🔄 Reconnecting in ${delay / 1000}s...`);

    reconnectTimerRef.current = setTimeout(() => {
      reconnectDelayRef.current = Math.min(
        delay * 1.5,
        MAX_RECONNECT_DELAY
      );
      connect();
    }, delay);
  }, [connect]);

  // ── Connect on mount, cleanup on unmount ──
  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return (
    <div className="w-full">
      {/* ── Connection Status Bar ── */}
      <div className="flex items-center justify-between mb-4 px-1">
        <div className="flex items-center gap-2">
          <div
            className={`status-dot ${isConnected ? "active" : "inactive"}`}
          />
          <span className="font-mono text-xs text-gray-400">
            {isConnected ? "STREAM ACTIVE" : "CONNECTING..."}
          </span>
        </div>
        <span className="font-mono text-xs text-gray-500">
          ws://localhost:8766
        </span>
      </div>

      {/* ── Side-by-Side Video Panels ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 lg:gap-6">
        {/* ── LR Panel (Input) ── */}
        <div className="glass-card halftone-overlay p-4 lg:p-5">
          {/* ── Panel Header ── */}
          <div className="flex items-center justify-between mb-3 relative z-10">
            <div className="flex items-center gap-2.5">
              <div className="w-2.5 h-2.5 rounded-sm bg-neon-magenta opacity-80" />
              <h3 className="font-display font-semibold text-sm tracking-wide text-gray-300 uppercase">
                Low Resolution Input
              </h3>
            </div>
            <span className="metric-badge text-neon-magenta text-xs">
              {lrResolution}
            </span>
          </div>

          {/* ── Canvas Container ── */}
          <div className="relative rounded-xl overflow-hidden bg-noir-950 border border-noir-600/30">
            <canvas
              ref={lrCanvasRef}
              className="video-canvas w-full h-auto block"
              style={{ aspectRatio: "16/9" }}
            />

            {/* ── No Signal Overlay ── */}
            {!isConnected && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-noir-950/90 scan-lines">
                <div className="text-neon-magenta font-mono text-lg font-bold mb-2 animate-pulse">
                  NO SIGNAL
                </div>
                <div className="text-gray-500 font-mono text-xs">
                  Waiting for Host Server...
                </div>
              </div>
            )}

            {/* ── LR Badge ── */}
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-noir-950/80 border border-neon-magenta/30 font-mono text-xs text-neon-magenta">
              LR
            </div>
          </div>
        </div>

        {/* ── HR Panel (AI Upscaled Output) ── */}
        <div className="glass-card halftone-overlay p-4 lg:p-5">
          {/* ── Panel Header ── */}
          <div className="flex items-center justify-between mb-3 relative z-10">
            <div className="flex items-center gap-2.5">
              <div className="w-2.5 h-2.5 rounded-sm bg-neon-cyan opacity-80" />
              <h3 className="font-display font-semibold text-sm tracking-wide text-gray-300 uppercase">
                AI Super Resolution
              </h3>
            </div>
            <span className="metric-badge text-neon-cyan text-xs">
              {hrResolution}
            </span>
          </div>

          {/* ── Canvas Container ── */}
          <div className="relative rounded-xl overflow-hidden bg-noir-950 border border-noir-600/30">
            <canvas
              ref={hrCanvasRef}
              className="video-canvas w-full h-auto block"
              style={{ aspectRatio: "16/9" }}
            />

            {/* ── No Signal Overlay ── */}
            {!isConnected && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-noir-950/90 scan-lines">
                <div className="text-neon-cyan font-mono text-lg font-bold mb-2 animate-pulse">
                  STANDBY
                </div>
                <div className="text-gray-500 font-mono text-xs">
                  AI Engine initializing...
                </div>
              </div>
            )}

            {/* ── HR Badge ── */}
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded-md bg-noir-950/80 border border-neon-cyan/30 font-mono text-xs text-neon-cyan">
              HR
            </div>

            {/* ── AI Badge ── */}
            <div className="absolute top-3 right-3 px-2.5 py-1 rounded-md bg-neon-cyan/10 border border-neon-cyan/20 font-mono text-xs text-neon-cyan flex items-center gap-1.5">
              <svg
                className="w-3 h-3"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 10V3L4 14h7v7l9-11h-7z"
                />
              </svg>
              AI Enhanced
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
