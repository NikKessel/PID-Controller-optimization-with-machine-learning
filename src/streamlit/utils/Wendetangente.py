import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

class ZieglerNicholsTuner:
    def __init__(self):
        self.L = None  # Dead time from Wendetangenten
        self.tau = None  # Time constant from Wendetangenten
        self.K = None  # Process gain
        self.inflection_point = None
        self.tangent_params = None
        
    def create_pt2_system(self, K, T1, T2, deadtime=0):
        """Create PT2 transfer function"""
        # PT2: K / ((T1*s + 1)(T2*s + 1))
        num = [K]
        den = np.convolve([T1, 1], [T2, 1])  # (T1*s + 1)(T2*s + 1)
        
        sys = signal.TransferFunction(num, den)
        return sys, deadtime  # Return system and deadtime separately
    
    def create_it1_system(self, K, T1, deadtime=0):
        """Create IT1 (integrating) transfer function"""
        # IT1: K / (s * (T1*s + 1))
        num = [K]
        den = [T1, 1, 0]  # s * (T1*s + 1) = T1*s^2 + s
        
        sys = signal.TransferFunction(num, den)
        return sys, deadtime  # Return system and deadtime separately
    
    def simulate_step_response(self, sys_tuple, t_final=50, n_points=2000):
        """Simulate step response with dead time"""
        sys, deadtime = sys_tuple
        t = np.linspace(0, t_final, n_points)
        
        # Get step response without dead time
        t_temp, y_temp = signal.step(sys, T=t)
        
        # Apply dead time by shifting the response
        if deadtime > 0:
            # Find the index corresponding to dead time
            deadtime_idx = int(deadtime / (t[1] - t[0]))
            y = np.zeros_like(t_temp)
            
            # Shift response by dead time
            if deadtime_idx < len(y):
                y[deadtime_idx:] = y_temp[:-deadtime_idx] if deadtime_idx > 0 else y_temp
        else:
            y = y_temp
            
        return t_temp, y
    
    def find_inflection_point(self, t, y):
        """Find inflection point (maximum slope) in step response"""
        # Calculate first derivative (slope)
        dt = t[1] - t[0]
        dy_dt = np.gradient(y, dt)
        
        # Find maximum slope point
        max_slope_idx = np.argmax(dy_dt)
        
        self.inflection_point = {
            't': t[max_slope_idx],
            'y': y[max_slope_idx], 
            'slope': dy_dt[max_slope_idx]
        }
        
        return self.inflection_point
    
    def fit_tangent_line(self, t, y):
        """Fit tangent line at inflection point"""
        ip = self.inflection_point
        
        # Tangent line: y = slope * (t - t_inflection) + y_inflection
        def tangent(time):
            return ip['slope'] * (time - ip['t']) + ip['y']
        
        self.tangent_params = {
            'slope': ip['slope'],
            'intercept': ip['y'] - ip['slope'] * ip['t']
        }
        
        return tangent
    
    def extract_wendetangenten_parameters(self, t, y, system_type='PT2'):
        """Extract L (dead time) and tau (time constant) using Wendetangenten method"""
        tangent_func = self.fit_tangent_line(t, y)
        
        if system_type == 'PT2':
            # For PT2: find where tangent intersects initial value (y=0) and final value
            y_initial = 0
            y_final = y[-1]  # Steady state value
            
            # Dead time L: where tangent line intersects y = y_initial
            # 0 = slope * (t - t_inflection) + y_inflection
            t_dead = self.inflection_point['t'] - self.inflection_point['y'] / self.inflection_point['slope']
            self.L = max(0, t_dead)  # Ensure positive
            
            # Time constant tau: time from end of dead time to 63% of final value
            # Find where tangent intersects y = y_final
            t_final_tangent = (y_final - self.inflection_point['y']) / self.inflection_point['slope'] + self.inflection_point['t']
            
            self.tau = t_final_tangent - self.L
            self.K = y_final  # Process gain
            
        elif system_type == 'IT1':
            # For integrating systems, use different approach
            # Use the slope at inflection point and time to characterize
            self.K = self.inflection_point['slope']  # For IT1, gain relates to slope
            
            # For IT1, estimate parameters from response shape
            # This is a simplified approach - more complex methods exist
            t_63 = None
            y_max_slope = self.inflection_point['y']
            
            # Find time where response reaches certain multiple of inflection point value
            target_y = 2 * y_max_slope
            idx_63 = np.where(y >= target_y)[0]
            if len(idx_63) > 0:
                t_63 = t[idx_63[0]]
                self.tau = t_63 - self.inflection_point['t']
                self.L = self.inflection_point['t'] / 2  # Rough estimate
            else:
                self.tau = self.inflection_point['t']
                self.L = self.inflection_point['t'] / 3
    
    def calculate_pid_parameters(self, method='ZN'):
        """Calculate PID parameters using official formulas from tables"""
        if self.L is None or self.tau is None:
            raise ValueError("Must extract Wendetangenten parameters first")
        
        # Using the exact formulas from the official tables
        # Where Tg = self.tau (time constant from Wendetangenten)
        # Where Tu = self.L (dead time from Wendetangenten) 
        # Where Ks = self.K (process gain)
        
        Tg = self.tau  # Tg in formulas
        Tu = self.L    # Tu in formulas  
        Ks = self.K    # Ks in formulas
        
        if method == 'ZN':
            # Ziegler-Nichols Method 2: Step Response (from Image 1)
            # PID row: Kp = 1.2*Tg/(Ks*Tu), TN = 2*Tu, TV = 0.5*Tu
            Kp = 1.2 * Tg / (Ks * Tu) if Tu > 0 else 0.5 / Ks
            TN = 2 * Tu if Tu > 0 else Tg  # TN is integral time (Ti)
            TV = 0.5 * Tu if Tu > 0 else 0  # TV is derivative time (Td)
            
        elif method == 'CHR_aperiodic':
            # CHR Aperiodic (0% overshoot) from Image 2 - left side
            # PID row: Kp = 0.6*Tg/(Ks*Tu), TN = 1*Tg, TV = 0.5*Tu
            Kp = 0.6 * Tg / (Ks * Tu) if Tu > 0 else 0.3 / Ks
            TN = 1.0 * Tg if Tu > 0 else 2 * Tg
            TV = 0.5 * Tu if Tu > 0 else 0
            
        elif method == 'CHR_20':
            # CHR 20% Overshoot from Image 2 - right side  
            # PID row: Kp = 0.95*Tg/(Ks*Tu), TN = 1.35*Tg, TV = 0.47*Tu
            Kp = 0.95 * Tg / (Ks * Tu) if Tu > 0 else 0.7 / Ks
            TN = 1.35 * Tg if Tu > 0 else Tg
            TV = 0.47 * Tu if Tu > 0 else 0
            
        # Convert to standard PID form
        Ti = TN  # Integral time
        Td = TV  # Derivative time
        Ki = Kp / Ti if Ti > 0 else 0
        Kd = Kp * Td
        
        return {
            'Kp': Kp,
            'Ki': Ki, 
            'Kd': Kd,
            'Ti': Ti,
            'Td': Td,
            'TN': TN,
            'TV': TV,
            'L': self.L,
            'tau': self.tau,
            'K': self.K
        }
    
    def plot_analysis(self, t, y, tangent_func, pid_params, system_type='PT2'):
        """Plot step response with Wendetangenten analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Step response with tangent line
        ax1.plot(t, y, 'b-', linewidth=2, label='Step Response')
        
        # Plot tangent line
        t_tangent = np.linspace(max(0, self.L - self.tau), self.L + 2*self.tau, 100)
        y_tangent = tangent_func(t_tangent)
        ax1.plot(t_tangent, y_tangent, 'r--', linewidth=2, label='Tangent Line')
        
        # Mark inflection point
        ip = self.inflection_point
        ax1.plot(ip['t'], ip['y'], 'ro', markersize=8, label=f"Inflection Point ({ip['t']:.2f}, {ip['y']:.2f})")
        
        # Mark dead time and time constant
        if system_type == 'PT2':
            ax1.axvline(x=self.L, color='g', linestyle=':', label=f'Dead Time L = {self.L:.2f}')
            ax1.axvline(x=self.L + self.tau, color='orange', linestyle=':', label=f'L + τ = {self.L + self.tau:.2f}')
            ax1.axhline(y=self.K, color='purple', linestyle=':', alpha=0.7, label=f'Steady State K = {self.K:.2f}')
        
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Output')
        ax1.set_title(f'{system_type} System Step Response with Wendetangenten Analysis')
        ax1.legend()
        
        # Plot 2: Derivative (slope)
        dt = t[1] - t[0]
        dy_dt = np.gradient(y, dt)
        ax2.plot(t, dy_dt, 'b-', linewidth=2, label='dy/dt (Slope)')
        ax2.axvline(x=ip['t'], color='r', linestyle='--', label=f"Max Slope at t = {ip['t']:.2f}")
        ax2.plot(ip['t'], ip['slope'], 'ro', markersize=8, label=f"Max Slope = {ip['slope']:.3f}")
        
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Slope')
        ax2.set_title('Step Response Derivative')
        ax2.legend()
        
        plt.tight_layout()
        
        # Print results
        print("=== Wendetangenten Analysis Results ===")
        print(f"System Type: {system_type}")
        print(f"Process Gain (K): {self.K:.4f}")
        print(f"Dead Time (L): {self.L:.4f} s")
        print(f"Time Constant (τ): {self.tau:.4f} s")
        print(f"Inflection Point: t = {ip['t']:.4f} s, y = {ip['y']:.4f}")
        print(f"Maximum Slope: {ip['slope']:.4f}")
        print()
        print("=== PID Parameters (Using Official Formulas) ===")
        for method, params in pid_params.items():
            print(f"{method}:")
            print(f"  Kp = {params['Kp']:.4f}")
            print(f"  Ki = {params['Ki']:.4f}")
            print(f"  Kd = {params['Kd']:.4f}")
            print(f"  TN (Ti) = {params['TN']:.4f} s")
            print(f"  TV (Td) = {params['TV']:.4f} s")
            print()
        
        return fig

def main():
    # Example usage
    tuner = ZieglerNicholsTuner()
    
    # Example 1: PT2 System
    print("=== PT2 System Example ===")
    K, T1, T2, deadtime = 2.0, 5.0, 2.0, 1.0
    
    # Create system and simulate
    sys_pt2 = tuner.create_pt2_system(K, T1, T2, deadtime)
    t, y = tuner.simulate_step_response(sys_pt2, t_final=30)
    
    # Apply Wendetangenten method
    tuner.find_inflection_point(t, y)
    tangent_func = tuner.fit_tangent_line(t, y)
    tuner.extract_wendetangenten_parameters(t, y, 'PT2')
    
    # Calculate PID parameters with different methods
    pid_params = {
        'Ziegler-Nichols (Method 2)': tuner.calculate_pid_parameters('ZN'),
        'CHR Aperiodic (0% Overshoot)': tuner.calculate_pid_parameters('CHR_aperiodic'),
        'CHR 20% Overshoot': tuner.calculate_pid_parameters('CHR_20')
    }
    
    # Plot results
    fig1 = tuner.plot_analysis(t, y, tangent_func, pid_params, 'PT2')
    plt.show()
    
    # Example 2: IT1 System
    print("\n" + "="*50)
    print("=== IT1 (Integrating) System Example ===")
    tuner2 = ZieglerNicholsTuner()
    
    K_int, T1_int, deadtime_int = 1.0, 3.0, 0.5
    
    # Create integrating system and simulate
    sys_it1 = tuner2.create_it1_system(K_int, T1_int, deadtime_int)
    t2, y2 = tuner2.simulate_step_response(sys_it1, t_final=20)
    
    # Apply Wendetangenten method for integrating system
    tuner2.find_inflection_point(t2, y2)
    tangent_func2 = tuner2.fit_tangent_line(t2, y2)
    tuner2.extract_wendetangenten_parameters(t2, y2, 'IT1')
    
    # For integrating systems, use conservative tuning
    pid_params2 = {
        'Modified ZN (IT1)': tuner2.calculate_pid_parameters('ZN')
    }
    
    # Plot results
    fig2 = tuner2.plot_analysis(t2, y2, tangent_func2, pid_params2, 'IT1')
    plt.show()

if __name__ == "__main__":
    main()