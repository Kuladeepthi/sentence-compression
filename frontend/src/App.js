import { useState } from "react";

const API_URL = "https://sentence-compression.onrender.com";

function App() {
  const [sentence, setSentence] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleCompress = async () => {
    if (!sentence.trim()) {
      setError("Please enter a sentence!");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${API_URL}/compress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence }),
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError("Could not connect to API. Try again!");
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>✂️ Sentence Compressor</h1>
      <p style={styles.subtitle}>Enter a sentence to compress it using NLP</p>

      <textarea
        style={styles.textarea}
        rows={4}
        placeholder="e.g. The cat that was sitting on the mat looked very hungry."
        value={sentence}
        onChange={(e) => setSentence(e.target.value)}
      />

      <button
        style={styles.button}
        onClick={handleCompress}
        disabled={loading}
      >
        {loading ? "Compressing..." : "Compress"}
      </button>

      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.resultBox}>
          <div style={styles.card}>
            <h3 style={styles.cardTitle}>Original</h3>
            <p style={styles.cardText}>{result.original}</p>
            <span style={styles.badge}>{result.original_word_count} words</span>
          </div>

          <div style={styles.arrow}>→</div>

          <div style={styles.card}>
            <h3 style={styles.cardTitle}>Compressed</h3>
            <p style={styles.cardText}>{result.compressed}</p>
            <span style={styles.badge}>{result.compressed_word_count} words</span>
          </div>

          <div style={styles.ratioBox}>
            Compression Ratio: <strong>{result.compression_ratio}</strong>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { maxWidth: 800, margin: "0 auto", padding: 40, fontFamily: "Arial, sans-serif" },
  title: { textAlign: "center", fontSize: 32, color: "#2d2d2d" },
  subtitle: { textAlign: "center", color: "#666", marginBottom: 30 },
  textarea: { width: "100%", padding: 12, fontSize: 16, borderRadius: 8, border: "1px solid #ccc", boxSizing: "border-box" },
  button: { marginTop: 12, width: "100%", padding: 14, fontSize: 18, backgroundColor: "#4f46e5", color: "white", border: "none", borderRadius: 8, cursor: "pointer" },
  error: { color: "red", marginTop: 10 },
  resultBox: { marginTop: 30, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" },
  card: { flex: 1, minWidth: 200, padding: 20, borderRadius: 10, backgroundColor: "#f9f9f9", border: "1px solid #ddd" },
  cardTitle: { margin: 0, marginBottom: 8, color: "#4f46e5" },
  cardText: { fontSize: 16, color: "#333" },
  badge: { display: "inline-block", marginTop: 8, padding: "4px 10px", backgroundColor: "#4f46e5", color: "white", borderRadius: 20, fontSize: 12 },
  arrow: { fontSize: 30, color: "#4f46e5" },
  ratioBox: { width: "100%", textAlign: "center", padding: 12, backgroundColor: "#eef2ff", borderRadius: 8, fontSize: 16 },
};

export default App;