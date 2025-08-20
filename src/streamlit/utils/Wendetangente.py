import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# -------------------------- Utility: polynomial ops (kept) ------------------- #
def poly_add(a, b):
    if len(a) < len(b):
        a = np.pad(a, (len(b) - len(a), 0))
    elif len(b) < len(a):
        b = np.pad(b, (len(a) - len(b), 0))
    return a + b

def tf_series(num1, den1, num2, den2):
    return np.polymul(num1, num2), np.polymul(den1, den2)

def tf_feedback(num, den):
    den_cl = poly_add(den, num)  # unity negative feedback: L/(1+L)
    num_cl = num
    return num_cl, den_cl


class ZieglerNicholsTuner:
    def __init__(self):
        self.L = None      # Tu (dead time) from Wendetangente
        self.tau = None    # Tg (time constant)
        self.K = None      # Ks (DC gain for unit step)
        self.inflection_point = None
        self.tangent_params = None

    # ---------------------- Plant definitions (open loop) --------------------- #
    def create_pt2_system(self, K, T1, T2, deadtime=0.0):
        """PT2: K / ((T1 s + 1)(T2 s + 1))."""
        num = [K]
        den = np.convolve([T1, 1.0], [T2, 1.0])
        sys = signal.TransferFunction(num, den)
        return sys, float(deadtime)

    def create_it1_system(self, K, T1, deadtime=0.0):
        """IT1: K / (s (T1 s + 1))."""
        num = [K]
        den = [T1, 1.0, 0.0]  # T1 s^2 + s
        sys = signal.TransferFunction(num, den)
        return sys, float(deadtime)

    def simulate_step_response(self, sys_tuple, t_final=50.0, n_points=2000):
        """Open-loop step response, dead time applied by shifting the output."""
        sys, deadtime = sys_tuple
        t = np.linspace(0.0, t_final, n_points)
        t0, y0 = signal.step(sys, T=t)

        if deadtime > 0:
            dt = t[1] - t[0]
            shift = int(round(deadtime / dt))
            y = np.zeros_like(y0)
            if shift < len(y0):
                y[shift:] = y0[:len(y0) - shift]
        else:
            y = y0
        return t, y

    # -------------------- Wendetangente: feature extraction ------------------- #
    def find_inflection_point(self, t, y):
        dt = t[1] - t[0]
        dy_dt = np.gradient(y, dt)
        i = int(np.argmax(dy_dt))
        self.inflection_point = {'t': t[i], 'y': y[i], 'slope': dy_dt[i]}
        return self.inflection_point

    def fit_tangent_line(self, t, y):
        ip = self.inflection_point
        self.tangent_params = {
            'slope': ip['slope'],
            'intercept': ip['y'] - ip['slope'] * ip['t']
        }
        def tangent(tt):
            return self.tangent_params['slope'] * tt + self.tangent_params['intercept']
        return tangent

    def extract_wendetangenten_parameters(self, t, y, system_type='PT2'):
        tangent = self.fit_tangent_line(t, y)
        y_final = float(y[-1])                     # steady-state value for unit step
        Tu = self.inflection_point['t'] - self.inflection_point['y'] / self.inflection_point['slope']
        Tu = max(0.0, Tu)
        t_cross_final = (y_final - self.inflection_point['y'])/self.inflection_point['slope'] + self.inflection_point['t']
        Tg = max(0.0, t_cross_final - Tu)
        self.L, self.tau, self.K = Tu, Tg, y_final   # Ks = steady value for unit step
        return self.L, self.tau, self.K

    # ----------------------------- PID formulas ------------------------------ #
    def calculate_pid_parameters(self, method='ZN'):
        if self.L is None or self.tau is None or self.K is None:
            raise ValueError("Run Wendetangente extraction first.")
        Tu, Tg, Ks = self.L, self.tau, self.K

        if method == 'ZN':
            Kp = 1.2 * Tg / (Ks * Tu) if Tu > 0 else 0.5 / Ks
            Ti = 2.0 * Tu if Tu > 0 else Tg
            Td = 0.5 * Tu if Tu > 0 else 0.0
        elif method == 'CHR_aperiodic':
            Kp = 0.6 * Tg / (Ks * Tu) if Tu > 0 else 0.3 / Ks
            Ti = 1.0 * Tg if Tu > 0 else 2.0 * Tg
            Td = 0.5 * Tu if Tu > 0 else 0.0
        elif method == 'CHR_20':
            Kp = 0.95 * Tg / (Ks * Tu) if Tu > 0 else 0.7 / Ks
            Ti = 1.35 * Tg if Tu > 0 else Tg
            Td = 0.47 * Tu if Tu > 0 else 0.0
        else:
            raise ValueError("Unknown method")

        Ki = Kp / Ti if Ti > 0 else 0.0
        Kd = Kp * Td
        return {'Kp': Kp, 'Ki': Ki, 'Kd': Kd, 'Ti': Ti, 'Td': Td,
                'Tu': Tu, 'Tg': Tg, 'Ks': Ks}

    # -------------------- Closed-loop sim with REAL time delay ---------------- #
    def simulate_closed_loop_td(self, K, T1, T2, L, Kp, Ki, Kd,
                               N=20.0, T_end=30.0, dt=0.002):
        """
        Simulate closed loop with explicit time delay via FIFO buffer.
        Plant: two cascaded 1st-order lags -> y = K * x2
            x1' = (-x1 + u_L)/T1
            x2' = (-x2 + x1)/T2
        Controller (parallel, filtered D): u = Kp*e + I + v
            I' = Ki*e
            v' = N*(Kd*de/dt - v)
        """
        n = int(T_end / dt) + 1
        t = np.linspace(0.0, T_end, n)

        # Delay buffer for plant input
        dsteps = max(1, int(round(L / dt))) if L > 0 else 0
        buf = np.zeros(dsteps + 1)

        # States
        x1 = x2 = y = 0.0
        I = 0.0
        v = 0.0
        e_prev = 0.0
        y_hist = np.zeros(n)

        for i, ti in enumerate(t):
            r = 1.0
            e = r - y
            de = (e - e_prev) / dt if i > 0 else 0.0

            # Controller
            v += dt * (N * (Kd * de - v))  # filtered derivative contribution
            I += dt * (Ki * e)             # integral
            u = Kp * e + I + v

            # Apply input delay to plant
            if dsteps > 0:
                buf[1:] = buf[:-1]
                buf[0] = u
                u_del = buf[-1]
            else:
                u_del = u

            # Plant (two 1st-order in series)
            x1 += dt * ((-x1 + u_del) / T1)
            x2 += dt * ((-x2 + x1) / T2)
            y = K * x2

            y_hist[i] = y
            e_prev = e

        return t, y_hist

    # ------------------------------- Plotting -------------------------------- #
    def _centered_text(self, ax, x, y, text, **kw):
        ax.annotate(text, (x, y), xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", **kw)

    def plot_analysis(self, t, y, tangent_func, pid_params_dict, system_type='PT2'):
        """Open-loop step + tangent + Tu/Tg/Ks and derivative panel."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Step
        ax1.plot(t, y, linewidth=2, label='Step Response')

        # Tangent
        t_tan = np.linspace(max(0, self.L - self.tau), self.L + 2*self.tau, 300)
        ax1.plot(t_tan, tangent_func(t_tan), '--', linewidth=2, label='Tangent Line')

        # Inflection
        ip = self.inflection_point
        ax1.plot(ip['t'], ip['y'], 'o', markersize=8, label=f'Inflection ({ip["t"]:.2f}, {ip["y"]:.2f})')

        # Tu/Tg/Ks visuals
        ax1.axvline(self.L, linestyle=':', label=f'Tu = {self.L:.2f}')
        ax1.axvline(self.L + self.tau, linestyle=':', label=f'Tu+Tg = {self.L + self.tau:.2f}')
        ax1.axhline(self.K, linestyle=':', alpha=0.8, label=f'Ks = {self.K:.2f}')
        ax1.axvspan(0, self.L, alpha=0.10)
        ax1.axvspan(self.L, self.L + self.tau, alpha=0.10)
        ytxt = 0.05 * (np.nanmax(y) - np.nanmin(y)) + np.nanmin(y)
        if self.L > 0:
            self._centered_text(ax1, self.L/2, ytxt, "Tu")
        self._centered_text(ax1, self.L + self.tau/2, ytxt, "Tg")
        # Compute tangent values at Tu and Tu+Tg
        y_Tu = tangent_func(self.L)
        y_Tg = tangent_func(self.L + self.tau)

        # Draw markers
        ax1.plot(self.L, y_Tu, 'ks', markersize=8, label='@Tu')
        ax1.plot(self.L + self.tau, y_Tg, 'ks', markersize=8, label='@Tu+Tg')

        # Annotate
        ax1.annotate("Tu", (self.L, y_Tu), textcoords="offset points", xytext=(0,10), ha='center')
        ax1.annotate("Tu+Tg", (self.L + self.tau, y_Tg), textcoords="offset points", xytext=(0,10), ha='center')
        # <<< END ADD >>>

        # Shaded spans for Tu and Tg
        ax1.axvspan(0, self.L, alpha=0.10)
        ax1.axvspan(self.L, self.L + self.tau, alpha=0.10)
        # Numeric box
        txt = f"Tu = {self.L:.3f} s\nTg = {self.tau:.3f} s\nKs = {self.K:.3f}"
        ax1.text(0.98, 0.02, txt, transform=ax1.transAxes,
                 ha='right', va='bottom', bbox=dict(boxstyle='round', alpha=0.15))

        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Output')
        ax1.set_title(f'{system_type} Step Response with Wendetangente (Tu, Tg, Ks)')
        ax1.legend(loc='best')

        # Derivative panel
        dt = t[1] - t[0]
        dy_dt = np.gradient(y, dt)
        ax2.plot(t, dy_dt, linewidth=2, label='dy/dt')
        ax2.axvline(ip['t'], linestyle='--', label=f"Max slope @ t={ip['t']:.2f}")
        ax2.plot(ip['t'], ip['slope'], 'o', label=f"Max slope = {ip['slope']:.3f}")
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Slope')
        ax2.set_title('Step Response Derivative')
        ax2.legend(loc='best')

        plt.tight_layout()
        return fig

    def plot_closed_loop_comparison(self, K, T1, T2, deadtime, pid_params_dict,
                                    T_end=30.0, dt=0.002, N=20.0):
        """Overlay closed-loop steps for ZN & CHR using explicit delay simulation."""
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, p in pid_params_dict.items():
            tt, yy = self.simulate_closed_loop_td(
                K, T1, T2, deadtime,
                p['Kp'], p['Ki'], p['Kd'],
                N=N, T_end=T_end, dt=dt
            )
            ax.plot(tt, yy, linewidth=2,
                    label=f"{name}  (Kp={p['Kp']:.3g}, Ki={p['Ki']:.3g}, Kd={p['Kd']:.3g})")

        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('y(t)')
        ax.axhline(1.0, color='k', alpha=0.2)  # reference line
        ax.set_title('Closed-Loop Step Responses (ZN & CHR) — explicit delay (no Padé)')
        ax.legend(loc='best')
        plt.tight_layout()
        return fig


# ------------------------------------ Demo ----------------------------------- #
def main():
    tuner = ZieglerNicholsTuner()

    # ---------- Example: PT2 plant ----------
    K, T1, T2, deadtime = 1.0, 2.0, 4.0, 0.0

    # Open-loop step (for Wendetangente extraction)
    sys_pt2 = tuner.create_pt2_system(K, T1, T2, deadtime)
    t, y = tuner.simulate_step_response(sys_pt2, t_final=30.0, n_points=2000)

    # Wendetangente
    tuner.find_inflection_point(t, y)
    tan = tuner.fit_tangent_line(t, y)
    Tu, Tg, Ks = tuner.extract_wendetangenten_parameters(t, y, 'PT2')

    # PID parameter sets
    pid_sets = {
        'Ziegler-Nichols': tuner.calculate_pid_parameters('ZN'),
        'CHR aperiodic (0%)': tuner.calculate_pid_parameters('CHR_aperiodic'),
        'CHR 20% overshoot': tuner.calculate_pid_parameters('CHR_20'),
    }

    # Plots
    tuner.plot_analysis(t, y, tan, pid_sets, 'PT2')
    tuner.plot_closed_loop_comparison(K, T1, T2, deadtime, pid_sets,
                                      T_end=30.0, dt=0.002, N=20.0)
    plt.show()


if __name__ == "__main__":
    main()
