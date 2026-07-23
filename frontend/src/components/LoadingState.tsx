type Props = {
  message?: string;
};

export function LoadingState({
  message = "Loading VoxForge insights…",
}: Props) {
  return (
    <div
      className="card flex items-center justify-center gap-3 p-8 text-center text-slate-300"
      role="status"
      aria-live="polite"
    >
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-forgeBlue border-t-transparent" />
      {message}
    </div>
  );
}
