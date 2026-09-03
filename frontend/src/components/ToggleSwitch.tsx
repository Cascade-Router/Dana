import "./ToggleSwitch.css";

type Props = {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
  /** "danger" recolors the active track/thumb red instead of green — for a
   * toggle whose "on" state is itself the risky one (e.g. Auto-Approve),
   * where green-for-on would send the wrong signal. */
  tone?: "default" | "danger";
};

// A real <input type="checkbox"> underneath (visually hidden, not removed)
// so keyboard nav (Tab/Space), screen readers, and form semantics all work
// natively — the pill/thumb are purely decorative siblings driven by the
// input's :checked state via CSS, never a click handler of their own.
// role="switch" relabels it for assistive tech as an on/off switch rather
// than a checkbox, matching what it actually represents here.
export function ToggleSwitch({ checked, onChange, label, disabled, tone = "default" }: Props) {
  return (
    <label
      className={`toggle-switch ${tone === "danger" ? "toggle-switch--danger" : ""} ${
        disabled ? "toggle-switch--disabled" : ""
      }`}
    >
      <input
        type="checkbox"
        role="switch"
        className="toggle-switch__input"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="toggle-switch__track" aria-hidden="true">
        <span className="toggle-switch__thumb" />
      </span>
      <span className="toggle-switch__label">{label}</span>
    </label>
  );
}
