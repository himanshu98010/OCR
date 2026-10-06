import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Braces,
  Check,
  ChevronDown,
  Clipboard,
  Download,
  FileImage,
  ImagePlus,
  Moon,
  RotateCcw,
  ScanText,
  Sparkles,
  Sun,
  X,
} from "lucide-react";
import nlp from "compromise";
import { createWorker, PSM } from "tesseract.js";

type View = "upload" | "ready" | "processing" | "result";
const acceptedTypes = ["image/png", "image/jpeg", "image/webp"];
const MAX_BYTES = 12 * 1024 * 1024;
const stages = [
  "Uploading",
  "Preparing image",
  "Recognizing text",
  "Almost done",
];

type TextInsights = {
  sentences: string[];
  nouns: string[];
  verbs: string[];
  keywords: string[];
};

/** Remove visual UI glyphs that OCR commonly mistakes for text in screenshots. */
function cleanOcrText(rawText: string) {
  return rawText
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) =>
      line
        // Stars, chevrons, and tiny interface icons often become these characters.
        .replace(/^\s*[<‹«»›]+\s*(?:\d+\s*)?(?=[A-Za-z])/u, "• ")
        .replace(/^\s*[~≈]+\s*$/u, "")
        // A lone trailing tilde is normally a missed UI ornament, not prose.
        .replace(/\s+[~≈]\s*$/u, "")
        .replace(/[ \t]{2,}/g, " ")
        .trimEnd(),
    )
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Performs local NLP tokenization, part-of-speech tagging, and keyword extraction. */
function analyseText(text: string): TextInsights {
  const document = nlp(text);
  const nounTerms = (document.nouns().out("array") as string[])
    .map((term) => term.trim())
    .filter(Boolean);
  const verbTerms = (document.verbs().out("array") as string[])
    .map((term) => term.trim())
    .filter(Boolean);
  const frequencies = nounTerms.reduce<Record<string, number>>((all, term) => {
    const normalized = term
      .toLowerCase()
      .replace(/[^a-z'-]/g, "")
      .trim();
    if (normalized.length > 2) all[normalized] = (all[normalized] || 0) + 1;
    return all;
  }, {});
  const keywords = Object.entries(frequencies)
    .sort(([, countA], [, countB]) => countB - countA)
    .slice(0, 5)
    .map(([term]) => term);
  const sentences = (document.sentences().out("array") as string[])
    .map((sentence) => sentence.trim())
    .filter(Boolean);

  return {
    sentences,
    nouns: [...new Set(nounTerms)],
    verbs: [...new Set(verbTerms)],
    keywords,
  };
}

/**
 * Enlarges and normalizes a photographed page, then separates two-column
 * layouts at their central whitespace. This keeps reading order intact.
 */
async function prepareForOcr(file: File): Promise<string[]> {
  const source = await createImageBitmap(file);
  const scale = Math.min(2, Math.max(1, 2200 / source.width));
  const width = Math.round(source.width * scale);
  const height = Math.round(source.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Image processing is unavailable");

  context.drawImage(source, 0, 0, width, height);
  source.close();
  const pixels = context.getImageData(0, 0, width, height);
  for (let index = 0; index < pixels.data.length; index += 4) {
    const grey =
      pixels.data[index] * 0.299 +
      pixels.data[index + 1] * 0.587 +
      pixels.data[index + 2] * 0.114;
    // Mild contrast enhancement improves photographed print without erasing fine strokes.
    const enhanced = Math.max(0, Math.min(255, (grey - 128) * 1.22 + 128));
    pixels.data[index] = enhanced;
    pixels.data[index + 1] = enhanced;
    pixels.data[index + 2] = enhanced;
  }
  context.putImageData(pixels, 0, 0);

  const darknessAt = (x: number) => {
    let ink = 0;
    let samples = 0;
    for (let y = Math.round(height * 0.12); y < height * 0.92; y += 10) {
      const value = pixels.data[(y * width + x) * 4];
      if (value < 165) ink += 1;
      samples += 1;
    }
    return ink / samples;
  };
  const rangeStart = Math.round(width * 0.38);
  const rangeEnd = Math.round(width * 0.62);
  const rawScores = Array.from({ length: rangeEnd - rangeStart }, (_, offset) =>
    darknessAt(rangeStart + offset),
  );
  // A real column gutter is a *wide* strip of whitespace. Averaging a 3%-wide
  // window prevents a narrow gap between words from being mistaken for a gutter.
  const windowRadius = Math.max(12, Math.round(width * 0.015));
  const scores = rawScores.map((_, index) => {
    const from = Math.max(0, index - windowRadius);
    const to = Math.min(rawScores.length, index + windowRadius);
    return (
      rawScores.slice(from, to).reduce((total, score) => total + score, 0) /
      (to - from)
    );
  });
  const averageInk =
    scores.reduce((total, score) => total + score, 0) / scores.length;
  const lowestInk = Math.min(...scores);
  const split = rangeStart + scores.indexOf(lowestInk);
  const hasColumnGutter = lowestInk < averageInk * 0.35;

  const crop = (left: number, cropWidth: number) => {
    const result = document.createElement("canvas");
    result.width = cropWidth;
    result.height = height;
    result
      .getContext("2d")!
      .drawImage(canvas, left, 0, cropWidth, height, 0, 0, cropWidth, height);
    return result.toDataURL("image/png");
  };

  if (!hasColumnGutter) return [canvas.toDataURL("image/png")];
  // Keep a small overlap so a slightly imperfect split never cuts characters.
  const overlap = Math.round(8 * scale);
  return [
    crop(0, split + overlap),
    crop(split - overlap, width - split + overlap),
  ];
}

function Button({
  children,
  onClick,
  kind = "primary",
  disabled = false,
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  kind?: "primary" | "secondary" | "quiet";
  disabled?: boolean;
  type?: "button" | "submit";
}) {
  return (
    <motion.button
      type={type}
      whileHover={disabled ? undefined : { y: -1 }}
      whileTap={disabled ? undefined : { scale: 0.98 }}
      className={`button ${kind}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </motion.button>
  );
}

export default function App() {
  const [darkMode, setDarkMode] = useState(
    () => localStorage.getItem("ocr-reader-theme") === "dark",
  );
  const [view, setView] = useState<View>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [stage, setStage] = useState(0);
  const [progress, setProgress] = useState(0);
  const [copied, setCopied] = useState(false);
  const [insightsOpen, setInsightsOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const insights = useMemo(
    () => (view === "result" ? analyseText(text) : null),
    [text, view],
  );

  useEffect(
    () => () => {
      if (preview) URL.revokeObjectURL(preview);
    },
    [preview],
  );
  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1800);
    return () => clearTimeout(timer);
  }, [copied]);
  useEffect(() => {
    document.documentElement.dataset.theme = darkMode ? "dark" : "light";
    localStorage.setItem("ocr-reader-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  const selectFile = useCallback(
    (next: File) => {
      setError("");
      if (!acceptedTypes.includes(next.type)) {
        setError("Choose a PNG, JPG, or WEBP image.");
        return;
      }
      if (next.size > MAX_BYTES) {
        setError("This image is larger than 12 MB. Choose a smaller file.");
        return;
      }
      const url = URL.createObjectURL(next);
      const image = new Image();
      image.onload = () => {
        if (preview) URL.revokeObjectURL(preview);
        setFile(next);
        setPreview(url);
        setView("ready");
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        setError("We couldn't open this image. Try another file.");
      };
      image.src = url;
    },
    [preview],
  );

  const process = async () => {
    if (!file) return;
    setView("processing");
    setStage(0);
    setProgress(0);
    setError("");
    try {
      const imageSections = await prepareForOcr(file);
      let activeSection = 0;
      const worker = await createWorker("eng", 1, {
        logger: (message) => {
          if (message.status === "loading tesseract core") setStage(1);
          if (message.status === "initializing tesseract") setStage(1);
          if (message.status === "recognizing text") {
            setStage(2);
            if (typeof message.progress === "number")
              setProgress(
                Math.round(
                  ((activeSection + message.progress) / imageSections.length) *
                    100,
                ),
              );
          }
        },
      });
      await worker.setParameters({ tessedit_pageseg_mode: PSM.SINGLE_COLUMN });
      const recognizedSections: string[] = [];
      for (
        activeSection = 0;
        activeSection < imageSections.length;
        activeSection += 1
      ) {
        const { data } = await worker.recognize(imageSections[activeSection]);
        recognizedSections.push(data.text);
      }
      await worker.terminate();
      setStage(3);
      setProgress(100);
      const output = cleanOcrText(recognizedSections.join("\n\n"));
      if (!output) {
        setError("Couldn't find readable text in this image.");
        setView("ready");
        return;
      }
      window.setTimeout(() => {
        setText(output);
        setInsightsOpen(false);
        setView("result");
      }, 350);
    } catch {
      setError("We couldn't read this image. Please try another one.");
      setView("ready");
    }
  };

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview("");
    setText("");
    setError("");
    setView("upload");
    setProgress(0);
  };
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
  };
  const download = () => {
    const url = URL.createObjectURL(
      new Blob([text], { type: "text/plain;charset=utf-8" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${file?.name.replace(/\.[^.]+$/, "") || "extracted-text"}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    const paste = (event: ClipboardEvent) => {
      const item = [...(event.clipboardData?.items || [])].find((i) =>
        i.type.startsWith("image/"),
      );
      if (item) {
        const pasted = item.getAsFile();
        if (pasted) selectFile(pasted);
      }
    };
    window.addEventListener("paste", paste);
    return () => window.removeEventListener("paste", paste);
  }, [selectFile]);

  return (
    <main>
      <header>
        <div className="brand">
          <span className="brand-mark">
            <ScanText size={17} />
          </span>
          <span>OCR Reader</span>
        </div>
        <div className="header-actions">
          <p>Extract text from images instantly.</p>
          <button
            className="theme-toggle"
            onClick={() => setDarkMode((value) => !value)}
            aria-label={darkMode ? "Use light mode" : "Use dark mode"}
            aria-pressed={darkMode}
          >
            {darkMode ? <Sun size={16} /> : <Moon size={16} />}
          </button>
        </div>
      </header>
      <section className="hero">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55 }}
          className="intro"
        >
          <h1>Words, made editable.</h1>
          <p>
            Upload an image and turn its text into something you can copy, save,
            and use.
          </p>
        </motion.div>
        <AnimatePresence mode="wait">
          {view === "upload" && (
            <motion.div
              key="upload"
              className={`upload-card ${dragging ? "dragging" : ""}`}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.3 }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                const dropped = e.dataTransfer.files[0];
                if (dropped) selectFile(dropped);
              }}
            >
              <div className="upload-icon">
                <ImagePlus size={24} />
              </div>
              <h2>Upload your image</h2>
              <p>Drag and drop, paste, or browse your files.</p>
              <Button onClick={() => fileInput.current?.click()}>
                <FileImage size={16} /> Upload image
              </Button>
              <span className="formats">PNG, JPG, or WEBP · up to 12 MB</span>
            </motion.div>
          )}
          {(view === "ready" || view === "processing" || view === "result") && (
            <motion.div
              key="workspace"
              className="workspace"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.32 }}
            >
              <div className="image-panel">
                <div className="panel-top">
                  <span>Original image</span>
                  {view !== "processing" && (
                    <button
                      className="icon-button"
                      onClick={reset}
                      aria-label="Remove image"
                    >
                      <X size={17} />
                    </button>
                  )}
                </div>
                <div className="image-wrap">
                  <img src={preview} alt="Selected for text extraction" />
                </div>
                <p className="file-name">{file?.name}</p>
              </div>
              <div className="output-panel">
                <AnimatePresence mode="wait">
                  {view === "ready" && (
                    <motion.div
                      key="ready"
                      className="action-state"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <div className="subtle-icon">
                        <Sparkles size={21} />
                      </div>
                      <h2>Ready to read</h2>
                      <p>
                        We’ll keep your image private. Processing happens right
                        here in your browser.
                      </p>
                      <Button onClick={process}>
                        <ScanText size={16} /> Extract text
                      </Button>
                    </motion.div>
                  )}
                  {view === "processing" && (
                    <motion.div
                      key="processing"
                      className="processing-state"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <div className="scan">
                        <span />
                        <span />
                        <span />
                      </div>
                      <div>
                        <span className="eyebrow">{stages[stage]}</span>
                        <h2>Reading your image</h2>
                        <p>Finding every readable word and line.</p>
                      </div>
                      <div className="progress-track" aria-label="OCR progress">
                        <motion.div
                          className="progress-fill"
                          initial={{ width: "10%" }}
                          animate={{
                            width: progress
                              ? `${Math.max(progress, 12)}%`
                              : ["12%", "68%", "34%"],
                          }}
                          transition={
                            progress
                              ? { duration: 0.35 }
                              : {
                                  duration: 2.4,
                                  repeat: Infinity,
                                  repeatType: "mirror",
                                }
                          }
                        />
                      </div>
                      {progress > 0 && (
                        <span className="progress-label">
                          {progress}% recognized
                        </span>
                      )}
                    </motion.div>
                  )}
                  {view === "result" && (
                    <motion.div
                      key="result"
                      className="result-state"
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.35 }}
                    >
                      <div className="result-head">
                        <div>
                          <span className="eyebrow success">
                            <Check size={13} /> Complete
                          </span>
                          <h2>Extracted text</h2>
                        </div>
                        <span className="word-count">
                          {text.trim() ? text.trim().split(/\s+/).length : 0}{" "}
                          words
                        </span>
                      </div>
                      <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        aria-label="Extracted text"
                      />
                      {insights && (
                        <section
                          className={`insights ${insightsOpen ? "expanded" : ""}`}
                          aria-label="NLP analysis"
                          role="button"
                          tabIndex={0}
                          aria-expanded={insightsOpen}
                          onClick={() => setInsightsOpen((value) => !value)}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              setInsightsOpen((value) => !value);
                            }
                          }}
                        >
                          <div className="insights-title">
                            <span>
                              <Braces size={14} /> NLP text insights
                            </span>
                            <small>
                              View details <ChevronDown size={14} />
                            </small>
                          </div>
                          <div className="insight-metrics">
                            <span>
                              <b>{insights.sentences.length}</b> sentences
                            </span>
                            <span>
                              <b>{insights.nouns.length}</b> nouns
                            </span>
                            <span>
                              <b>{insights.verbs.length}</b> verbs
                            </span>
                          </div>
                          {insights.keywords.length > 0 && (
                            <div className="keywords">
                              {insights.keywords.map((keyword) => (
                                <span key={keyword}>{keyword}</span>
                              ))}
                            </div>
                          )}
                        </section>
                      )}
                      <AnimatePresence>
                        {insights && insightsOpen && (
                          <motion.div
                            className="analysis-backdrop"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setInsightsOpen(false)}
                          >
                            <motion.section
                              className="analysis-modal"
                              role="dialog"
                              aria-modal="true"
                              aria-label="Detailed NLP analysis"
                              initial={{ opacity: 0, y: 14, scale: 0.98 }}
                              animate={{ opacity: 1, y: 0, scale: 1 }}
                              exit={{ opacity: 0, y: 10, scale: 0.98 }}
                              transition={{ duration: 0.2 }}
                              onClick={(event) => event.stopPropagation()}
                            >
                              <header className="modal-header">
                                <div>
                                  <span className="eyebrow success">
                                    <Braces size={13} /> NLP analysis
                                  </span>
                                  <h2>Language breakdown</h2>
                                  <p>
                                    Every item below was detected from the
                                    extracted text.
                                  </p>
                                </div>
                                <button
                                  className="modal-close"
                                  onClick={() => setInsightsOpen(false)}
                                  aria-label="Close NLP analysis"
                                >
                                  <X size={18} />
                                </button>
                              </header>
                              <div className="language-groups">
                                <article className="sentences-card">
                                  <h3>
                                    Sentences{" "}
                                    <small>Complete thoughts in the text</small>
                                  </h3>
                                  <ol className="sentence-list">
                                    {insights.sentences.map(
                                      (sentence, index) => (
                                        <li key={`${sentence}-${index}`}>
                                          {sentence}
                                        </li>
                                      ),
                                    )}
                                  </ol>
                                </article>
                                <article>
                                  <h3>
                                    Nouns{" "}
                                    <small>
                                      People, places, things, or ideas
                                    </small>
                                  </h3>
                                  <div className="term-list">
                                    {insights.nouns.map((noun, index) => (
                                      <span key={`${noun}-${index}`}>
                                        {noun}
                                      </span>
                                    ))}
                                  </div>
                                </article>
                                <article>
                                  <h3>
                                    Verbs{" "}
                                    <small>Actions or states of being</small>
                                  </h3>
                                  <div className="term-list">
                                    {insights.verbs.map((verb, index) => (
                                      <span key={`${verb}-${index}`}>
                                        {verb}
                                      </span>
                                    ))}
                                  </div>
                                </article>
                                <article>
                                  <h3>
                                    Keywords{" "}
                                    <small>Most frequent noun terms</small>
                                  </h3>
                                  <div className="term-list">
                                    {insights.keywords.map((keyword) => (
                                      <span key={keyword}>{keyword}</span>
                                    ))}
                                  </div>
                                </article>
                              </div>
                            </motion.section>
                          </motion.div>
                        )}
                      </AnimatePresence>
                      <div className="result-actions">
                        <Button kind="secondary" onClick={copy}>
                          {copied ? (
                            <Check size={16} />
                          ) : (
                            <Clipboard size={16} />
                          )}{" "}
                          {copied ? "Copied" : "Copy"}
                        </Button>
                        <Button kind="secondary" onClick={download}>
                          <Download size={16} /> Download
                        </Button>
                        <Button kind="quiet" onClick={reset}>
                          <RotateCcw size={15} /> Another image
                        </Button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        {error && (
          <motion.div
            role="alert"
            className="error"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {error}
            <button onClick={() => setError("")} aria-label="Dismiss error">
              <X size={16} />
            </button>
          </motion.div>
        )}
        <input
          ref={fileInput}
          className="sr-only"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(e) => {
            const chosen = e.target.files?.[0];
            if (chosen) selectFile(chosen);
            e.currentTarget.value = "";
          }}
        />
      </section>
      <footer>Designed for quiet, focused work.</footer>
    </main>
  );
}
