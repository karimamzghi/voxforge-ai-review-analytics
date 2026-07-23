type Props = {
  message: string;
  onRetry?: () => void;
};

export function ErrorState({ message, onRetry }: Props) {
  return (
    <div
      className="rounded-2xl border border-red-400/30 bg-red-500/10 p-6 text-red-100"
      role="alert"
    >
      <p className="font-semibold">Unable to load analytics</p>
      <p className="mt-1 text-sm text-red-200/80">{message}</p>

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-4 rounded-xl bg-red-200 px-4 py-2 text-sm font-semibold text-red-950 transition hover:bg-white"
        >
          Try again
        </button>
      )}
    </div>
  );
}
