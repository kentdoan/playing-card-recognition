import React from "react";
import { useCallback, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onDrop = useCallback((acceptedFiles) => {
    const selected = acceptedFiles?.[0];
    if (!selected) return;

    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  }, [previewUrl]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    multiple: false,
    accept: {
      "image/*": [".jpg", ".jpeg", ".png", ".webp"],
    },
  });

  const handleRecognize = async () => {
    if (!file || loading) return;

    setLoading(true);
    setResult(null);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/recognize?method=template`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Không thể nhận diện lá bài.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Đã có lỗi xảy ra. Vui lòng thử lại.");
    } finally {
      setLoading(false);
    }
  };

  const dropzoneClass = useMemo(() => {
    const base = "rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-200";
    if (isDragActive) {
      return `${base} border-cyan-400 bg-cyan-500/10 shadow-[0_0_40px_-12px_rgba(34,211,238,0.7)]`;
    }
    return `${base} border-zinc-600 bg-zinc-900/50 hover:border-cyan-500 hover:bg-zinc-900`;
  }, [isDragActive]);

  return (
    <main className="min-h-screen bg-linear-to-b from-zinc-950 via-zinc-900 to-black text-zinc-100">
      <div className="mx-auto w-full max-w-5xl px-4 py-10 sm:py-16">
        <header className="mb-10 text-center">
          <h1 className="text-3xl font-bold tracking-tight sm:text-5xl">Nhận diện lá bài</h1>
          <p className="mx-auto mt-4 max-w-2xl text-sm text-zinc-400 sm:text-base">
            Kéo thả ảnh vào khung bên dưới hoặc bấm để chọn file. Kết quả sẽ hiển thị ngay sau khi xử lý.
          </p>
        </header>

        <section className="rounded-3xl border border-zinc-800/90 bg-zinc-900/60 p-4 shadow-2xl backdrop-blur sm:p-6">
          <div {...getRootProps()} className={dropzoneClass}>
            <input {...getInputProps()} />
            <p className="text-base font-medium text-zinc-200">
              {isDragActive ? "Thả ảnh vào đây..." : "Kéo và thả ảnh vào đây, hoặc bấm để chọn file"}
            </p>
            <p className="mt-2 text-sm text-zinc-400">Hỗ trợ JPG, JPEG, PNG, WEBP</p>
          </div>

          <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center">
            <button
              type="button"
              onClick={handleRecognize}
              disabled={!file || loading}
              className="inline-flex items-center justify-center rounded-xl bg-cyan-500 px-5 py-3 font-semibold text-zinc-900 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <span className="inline-flex items-center gap-2">
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-900 border-t-transparent" />
                    Đang nhận diện...
                </span>
              ) : (
                "Nhận diện"
              )}
            </button>

            {file ? (
              <span className="text-sm text-zinc-400">File đã chọn: {file.name}</span>
            ) : (
              <span className="text-sm text-zinc-500">Chưa có file nào được chọn</span>
            )}
          </div>
        </section>

        {error && (
          <div className="mt-6 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-rose-200">
            <p className="text-sm font-medium">Lỗi: {error}</p>
          </div>
        )}

        {(previewUrl || result) && (
          <section className="mt-8 grid gap-6 lg:grid-cols-2">
            {previewUrl && (
              <article className="overflow-hidden rounded-3xl border border-zinc-800 bg-zinc-900/70 shadow-xl">
                <div className="border-b border-zinc-800 px-5 py-3 text-sm text-zinc-300">Ảnh vừa tải lên</div>
                <img src={previewUrl} alt="Ảnh lá bài đã tải lên" className="h-105 w-full object-contain bg-black/40 p-3" />
              </article>
            )}

            <article className="rounded-3xl border border-cyan-500/30 bg-linear-to-br from-cyan-500/10 to-zinc-900 p-6 shadow-[0_0_80px_-35px_rgba(34,211,238,0.8)]">
              <h2 className="text-lg font-semibold text-cyan-200">Kết quả nhận diện</h2>
              <div className="mt-4">
                {loading ? (
                  <div className="animate-pulse space-y-3">
                    <div className="h-5 w-2/3 rounded bg-zinc-700" />
                    <div className="h-5 w-1/2 rounded bg-zinc-700" />
                  </div>
                ) : result ? (
                  <>
                    <p className="text-3xl font-bold tracking-wide text-white">{result.result}</p>
                    <p className="mt-2 text-sm text-zinc-300">Thời gian xử lý: {result.processing_time}</p>
                  </>
                ) : (
                  <p className="text-zinc-400">Kết quả sẽ hiển thị tại đây sau khi xử lý.</p>
                )}
              </div>
            </article>
          </section>
        )}
      </div>
    </main>
  );
}

export default App;
