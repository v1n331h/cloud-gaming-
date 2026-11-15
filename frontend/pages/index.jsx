/**
 * ============================================
 * SuperStream AI — Main Dashboard
 * ============================================
 * The primary layout incorporating:
 * - Spider-Verse / Minh Pham aesthetic
 * - GSAP entrance animations
 * - VideoPlayer (side-by-side LR vs HR)
 * - TelemetryDashboard (real-time metrics)
 * - Animated background elements
 *
 * Design Language:
 * - Deep noir base (#08080c)
 * - Neon magenta (#ff2d7b) + cyan (#00f0ff) accents
 * - Halftone dot textures
 * - Glassmorphism panels
 * - Clean, bold typography (Space Grotesk + JetBrains Mono)
 * ============================================
 */

import Head from "next/head";
import { useEffect, useRef, useState, useCallback } from "react";
import dynamic from "next/dynamic";

// ── Dynamic imports (no SSR for WebSocket components) ──
const VideoPlayer = dynamic(() => import("@/components/VideoPlayer"), {
  ssr: false,
});
const TelemetryDashboard = dynamic(
  () => import("@/components/TelemetryDashboard"),
  { ssr: false }
);

export default function Home() {
  const [metrics, setMetrics] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [gsapLoaded, setGsapLoaded] = useState(false);

  // ── Refs for GSAP animations ──
  const heroRef = useRef(null);
  const videoSectionRef = useRef(null);
  const telemetrySectionRef = useRef(null);
  const footerRef = useRef(null);
  const orbLeftRef = useRef(null);
  const orbRightRef = useRef(null);

  // ── Metrics handler ──
  const handleMetrics = useCallback((newMetrics) => {
    setMetrics(newMetrics);
  }, []);

  // ── Connection status handler ──
  const handleConnection = useCallback((connected) => {
    setIsConnected(connected);
  }, []);

  // ── GSAP Entrance Animations ──
  useEffect(() => {
    let gsap, ScrollTrigger;

    async function initGSAP() {
      try {
        const gsapModule = await import("gsap");
        gsap = gsapModule.gsap || gsapModule.default;

        // ── Master Timeline for entrance sequence ──
        const tl = gsap.timeline({
          defaults: {
            ease: "power3.out",
            duration: 1,
          },
        });

        // ── Animate hero section ──
        tl.fromTo(
          heroRef.current,
          { opacity: 0, y: 40 },
          { opacity: 1, y: 0, duration: 0.8 }
        );

        // ── Animate hero children staggered ──
        const heroChildren = heroRef.current?.querySelectorAll("[data-animate]");
        if (heroChildren?.length) {
          tl.fromTo(
            heroChildren,
            { opacity: 0, y: 20 },
            { opacity: 1, y: 0, stagger: 0.12, duration: 0.6 },
            "-=0.4"
          );
        }

        // ── Animate video section ──
        tl.fromTo(
          videoSectionRef.current,
          { opacity: 0, y: 30, scale: 0.98 },
          { opacity: 1, y: 0, scale: 1, duration: 0.8 },
          "-=0.3"
        );

        // ── Animate telemetry cards ──
        tl.fromTo(
          telemetrySectionRef.current,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.7 },
          "-=0.4"
        );

        // ── Animate metric cards with stagger ──
        const metricCards = telemetrySectionRef.current?.querySelectorAll(
          '[id^="metric-"]'
        );
        if (metricCards?.length) {
          tl.fromTo(
            metricCards,
            { opacity: 0, y: 15, scale: 0.95 },
            { opacity: 1, y: 0, scale: 1, stagger: 0.1, duration: 0.5 },
            "-=0.3"
          );
        }

        // ── Animate footer ──
        tl.fromTo(
          footerRef.current,
          { opacity: 0 },
          { opacity: 1, duration: 0.5 },
          "-=0.2"
        );

        // ── Floating orb animations (continuous) ──
        if (orbLeftRef.current) {
          gsap.to(orbLeftRef.current, {
            y: -20,
            x: 10,
            duration: 6,
            ease: "sine.inOut",
            repeat: -1,
            yoyo: true,
          });
        }

        if (orbRightRef.current) {
          gsap.to(orbRightRef.current, {
            y: 15,
            x: -10,
            duration: 8,
            ease: "sine.inOut",
            repeat: -1,
            yoyo: true,
          });
        }

        setGsapLoaded(true);
      } catch (err) {
        console.warn("GSAP not available, skipping animations:", err);
        // ── Graceful fallback: make everything visible ──
        setGsapLoaded(true);
      }
    }

    initGSAP();
  }, []);

  return (
    <>
      <Head>
        <title>SuperStream AI — Real-Time AI Super Resolution</title>
        <meta
          name="description"
          content="Client-side AI super resolution streaming pipeline. Stream low, upscale with neural networks, experience high-fidelity video in real-time."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* ── Background Elements ── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
        {/* ── Grid overlay ── */}
        <div className="absolute inset-0 bg-grid-overlay opacity-40" />

        {/* ── Gradient orbs ── */}
        <div
          ref={orbLeftRef}
          className="absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full opacity-20"
          style={{
            background:
              "radial-gradient(circle, rgba(255,45,123,0.15) 0%, transparent 70%)",
          }}
        />
        <div
          ref={orbRightRef}
          className="absolute -bottom-32 -right-32 w-[600px] h-[600px] rounded-full opacity-15"
          style={{
            background:
              "radial-gradient(circle, rgba(0,240,255,0.12) 0%, transparent 70%)",
          }}
        />

        {/* ── Accent line ── */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-neon-magenta/50 to-transparent" />
      </div>

      {/* ── Main Content ── */}
      <main className="relative z-10 min-h-screen px-4 sm:px-6 lg:px-8 xl:px-12 py-6 lg:py-10 max-w-[1600px] mx-auto">
        {/* ═══════════════════════════════════════════
            HERO SECTION
            ═══════════════════════════════════════════ */}
        <header
          ref={heroRef}
          className="mb-10 lg:mb-14"
          style={{ opacity: gsapLoaded ? undefined : 0 }}
        >
          {/* ── Top Nav Bar ── */}
          <nav
            className="flex items-center justify-between mb-8 lg:mb-12"
            data-animate
          >
            {/* ── Logo ── */}
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-neon-magenta to-neon-purple flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M13 10V3L4 14h7v7l9-11h-7z"
                    />
                  </svg>
                </div>
                <div className="absolute -inset-1 rounded-xl bg-neon-magenta/20 blur-md -z-10" />
              </div>
              <span className="font-display font-bold text-lg tracking-tight">
                <span className="text-white">Super</span>
                <span className="text-neon-magenta">Stream</span>
              </span>
            </div>

            {/* ── Status Badge ── */}
            <div className="flex items-center gap-3" data-animate>
              <div
                className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-mono border ${
                  isConnected
                    ? "border-neon-green/30 text-neon-green bg-neon-green/5"
                    : "border-red-500/30 text-red-400 bg-red-500/5"
                }`}
              >
                <div
                  className={`w-1.5 h-1.5 rounded-full ${
                    isConnected
                      ? "bg-neon-green animate-pulse-glow"
                      : "bg-red-500"
                  }`}
                />
                {isConnected ? "PIPELINE ACTIVE" : "OFFLINE"}
              </div>
            </div>
          </nav>

          {/* ── Hero Title ── */}
          <div className="max-w-3xl" data-animate>
            <h1 className="font-display font-bold text-4xl sm:text-5xl lg:text-6xl xl:text-7xl tracking-tight leading-[1.05] mb-4 lg:mb-6">
              <span className="text-white">AI Super</span>
              <br />
              <span className="gradient-text-multi">Resolution</span>
              <br />
              <span className="text-white opacity-80">Pipeline</span>
            </h1>
          </div>

          {/* ── Subtitle ── */}
          <p
            className="font-body text-gray-500 text-base lg:text-lg max-w-xl leading-relaxed mb-6"
            data-animate
          >
            Stream video at <span className="text-neon-magenta font-semibold">360p</span>.
            Reconstruct at{" "}
            <span className="text-neon-cyan font-semibold">1080p</span> using a
            lightweight CNN. Save{" "}
            <span className="text-neon-yellow font-semibold">99%</span> bandwidth.
            In real-time.
          </p>

          {/* ── Tech Stack Pills ── */}
          <div className="flex flex-wrap gap-2" data-animate>
            {[
              { label: "PyTorch", color: "#EE4C2C" },
              { label: "FSRCNN", color: "#00f0ff" },
              { label: "WebSocket", color: "#b24dff" },
              { label: "Next.js", color: "#ffffff" },
              { label: "OpenCV", color: "#39ff14" },
            ].map(({ label, color }) => (
              <span
                key={label}
                className="px-3 py-1 rounded-full font-mono text-xs border transition-colors duration-200 hover:bg-white/5"
                style={{
                  color,
                  borderColor: `${color}25`,
                  backgroundColor: `${color}08`,
                }}
              >
                {label}
              </span>
            ))}
          </div>
        </header>

        {/* ═══════════════════════════════════════════
            VIDEO SECTION
            ═══════════════════════════════════════════ */}
        <section
          ref={videoSectionRef}
          className="mb-8 lg:mb-12"
          style={{ opacity: gsapLoaded ? undefined : 0 }}
        >
          <VideoPlayer
            onMetricsUpdate={handleMetrics}
            onConnectionChange={handleConnection}
          />
        </section>

        {/* ═══════════════════════════════════════════
            TELEMETRY SECTION
            ═══════════════════════════════════════════ */}
        <section
          ref={telemetrySectionRef}
          className="mb-10 lg:mb-14"
          style={{ opacity: gsapLoaded ? undefined : 0 }}
        >
          <TelemetryDashboard metrics={metrics} />
        </section>

        {/* ═══════════════════════════════════════════
            FOOTER
            ═══════════════════════════════════════════ */}
        <footer
          ref={footerRef}
          className="border-t border-noir-700/50 pt-6 pb-8"
          style={{ opacity: gsapLoaded ? undefined : 0 }}
        >
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            {/* ── Left: Project Info ── */}
            <div className="flex items-center gap-3">
              <span className="font-display font-semibold text-sm text-gray-500">
                SuperStream AI
              </span>
              <span className="text-gray-700">·</span>
              <span className="font-mono text-xs text-gray-600">
                Client-Side AI Super Resolution
              </span>
            </div>

            {/* ── Right: Architecture Legend ── */}
            <div className="flex items-center gap-4 font-mono text-xs text-gray-600">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-neon-magenta/60" />
                <span>Host Server</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-neon-purple/60" />
                <span>ML Engine</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-neon-cyan/60" />
                <span>Frontend</span>
              </div>
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
