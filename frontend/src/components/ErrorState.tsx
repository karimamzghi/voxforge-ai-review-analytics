type Props = { message: string };

export function ErrorState({ message }: Props) {
  return (
    <div className="rounded-2xl border border-red-400/30 bg-red-500/10 p-6 text-red-100">
      <p className="font-semibold">Unable to load analytics</p>
      <p className="mt-1 text-sm text-red-200/80">{message}</p>
    </div>
  );
}
