import "@/styles/globals.css";

/**
 * SuperStream AI — App Root
 *
 * Wraps all pages with the global Spider-Verse theme styles.
 * The noise-bg class adds a subtle film grain texture overlay.
 */
export default function App({ Component, pageProps }) {
  return (
    <div className="noise-bg min-h-screen">
      <Component {...pageProps} />
    </div>
  );
}
