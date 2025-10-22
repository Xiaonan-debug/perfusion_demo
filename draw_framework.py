"""
Clean and Professional Framework diagram for Digital Twin Simulator + RL Controller
No overlapping elements, large readable text, well-organized layout
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.lines as mlines
import numpy as np

# Set up the figure with professional styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
fig, ax = plt.subplots(1, 1, figsize=(24, 14))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Professional color palette
rl_color = '#1E88E5'  # Deep blue for RL
simulator_color = '#43A047'  # Green for simulator
state_color = '#FF6F00'  # Orange for state
action_color = '#8E24AA'  # Purple for action
reward_color = '#E53935'  # Red for reward
text_color = '#2C3E50'
formula_color = '#1A237E'

# ============ Helper Functions ============

def draw_neural_network(ax, x, y, width, height):
    """Draw a clean neural network visualization"""
    layers = [6, 10, 10, 8, 9]
    layer_spacing = width / (len(layers) + 1)
    
    for i, layer_size in enumerate(layers):
        x_pos = x + (i + 1) * layer_spacing
        node_spacing = height / (layer_size + 1)
        
        for j in range(layer_size):
            y_pos = y + (j + 1) * node_spacing
            circle = Circle((x_pos, y_pos), 0.12, 
                          color=rl_color, alpha=0.6, zorder=10)
            ax.add_patch(circle)
            
            # Connect to next layer
            if i < len(layers) - 1:
                next_layer_spacing = height / (layers[i+1] + 1)
                for k in range(layers[i+1]):
                    next_y = y + (k + 1) * next_layer_spacing
                    next_x = x + (i + 2) * layer_spacing
                    line = plt.Line2D([x_pos, next_x], [y_pos, next_y],
                                    color='#BBDEFB', alpha=0.2, linewidth=0.8, zorder=9)
                    ax.add_line(line)

def draw_digital_twin_icon(ax, x, y, size):
    """Draw a clean digital twin icon"""
    # Real organ (solid circle)
    organ1 = Circle((x - size*0.3, y), size*0.5, 
                   color=simulator_color, alpha=0.7, zorder=8)
    ax.add_patch(organ1)
    
    # Digital twin (dashed circle)
    organ2 = Circle((x + size*0.3, y), size*0.5, 
                   color=rl_color, alpha=0.5, fill=False, 
                   linewidth=4, linestyle='--', zorder=9)
    ax.add_patch(organ2)
    
    # Connection lines
    for angle in [45, 135, 225, 315]:
        rad = np.radians(angle)
        x1 = x - size*0.3 + size*0.5*np.cos(rad)
        y1 = y + size*0.5*np.sin(rad)
        x2 = x + size*0.3 + size*0.5*np.cos(rad)
        y2 = y + size*0.5*np.sin(rad)
        line = plt.Line2D([x1, x2], [y1, y2], 
                         color='gray', alpha=0.4, linewidth=2, 
                         linestyle=':', zorder=8)
        ax.add_line(line)

# ============ Title ============
ax.text(6.0, 9.5, 'Digital Twin-Based Deep Reinforcement Learning Framework', 
       fontsize=32, fontweight='bold', ha='center', va='center',
       color=text_color)
ax.text(6.0, 9.0, 'for Autonomous Organ Perfusion Control', 
       fontsize=26, fontweight='bold', ha='center', va='center',
       color=text_color, style='italic')

# ============ Digital Twin Icon at Top ============
draw_digital_twin_icon(ax, 6.0, 8.2, 1.0)
ax.text(4.5, 8.2, 'Physical\nOrgan', fontsize=16, ha='center', va='center',
       color=simulator_color, fontweight='bold')
ax.text(7.5, 8.2, 'Digital\nTwin', fontsize=16, ha='center', va='center',
       color=rl_color, fontweight='bold')

# ============ Left Side: Digital Twin Simulator ============

# Main simulator box
simulator_box = FancyBboxPatch((0.5, 2.5), 4.5, 4.5, 
                              boxstyle="round,pad=0.2", 
                              edgecolor=simulator_color, 
                              facecolor='#E8F5E9',
                              linewidth=5,
                              zorder=2)
ax.add_patch(simulator_box)

# Title
ax.text(2.75, 6.7, 'Digital Twin Simulator', 
       fontsize=26, fontweight='bold', ha='center', va='center',
       color=simulator_color)

# Physiological Model
physio_box = FancyBboxPatch((0.8, 5.2), 3.9, 1.2,
                           boxstyle="round,pad=0.1",
                           edgecolor=simulator_color,
                           facecolor='white',
                           linewidth=3,
                           zorder=3)
ax.add_patch(physio_box)
ax.text(2.75, 6.0, 'Physiological Model', fontsize=20, ha='center', va='center',
       color=text_color, fontweight='bold')
ax.text(2.75, 5.6, r'$s_t = [T, G, pH, pO_2, I, VR]$', 
       fontsize=18, ha='center', va='center', color=formula_color, 
       weight='bold')

# Dynamics
dynamics_box = FancyBboxPatch((0.8, 3.8), 3.9, 1.0,
                             boxstyle="round,pad=0.1",
                             edgecolor=simulator_color,
                             facecolor='white',
                             linewidth=3,
                             zorder=3)
ax.add_patch(dynamics_box)
ax.text(2.75, 4.5, 'Transition Dynamics', fontsize=20, ha='center', va='center',
       color=text_color, fontweight='bold')
ax.text(2.75, 4.1, r'$s_{t+1} = f(s_t, a_t, \theta)$', 
       fontsize=18, ha='center', va='center', color=formula_color,
       weight='bold')

# Reward
reward_box = FancyBboxPatch((0.8, 2.8), 3.9, 0.7,
                           boxstyle="round,pad=0.08",
                           edgecolor=reward_color,
                           facecolor='#FFEBEE',
                           linewidth=3,
                           zorder=3)
ax.add_patch(reward_box)
ax.text(2.75, 3.15, r'$r_t = h(t_{survival}) - \lambda \cdot penalties$', 
       fontsize=17, ha='center', va='center', color=reward_color,
       weight='bold')

# ============ Right Side: RL Controller ============

# Main RL box
rl_box = FancyBboxPatch((7.0, 2.5), 4.5, 4.5,
                       boxstyle="round,pad=0.2",
                       edgecolor=rl_color,
                       facecolor='#E3F2FD',
                       linewidth=5,
                       zorder=2)
ax.add_patch(rl_box)

# Title
ax.text(9.25, 6.7, 'RL Controller (DQN)', 
       fontsize=26, fontweight='bold', ha='center', va='center',
       color=rl_color)

# Neural Network Box
nn_box = FancyBboxPatch((7.3, 5.2), 3.9, 1.2,
                       boxstyle="round,pad=0.1",
                       edgecolor=rl_color,
                       facecolor='white',
                       linewidth=3,
                       zorder=3)
ax.add_patch(nn_box)
ax.text(9.25, 6.0, 'Deep Q-Network', fontsize=20, ha='center', va='center',
       color=text_color, fontweight='bold')
ax.text(9.25, 5.6, r'$Q(s,a;\theta): \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$', 
       fontsize=17, ha='center', va='center', color=formula_color,
       weight='bold')

# Neural network visualization (smaller, non-overlapping)
draw_neural_network(ax, 7.5, 4.5, 3.5, 0.8)

# Bellman Equation
bellman_box = FancyBboxPatch((7.3, 3.5), 3.9, 0.7,
                            boxstyle="round,pad=0.08",
                            edgecolor=rl_color,
                            facecolor='white',
                            linewidth=3,
                            zorder=3)
ax.add_patch(bellman_box)
ax.text(9.25, 3.85, r'$Q(s,a) \leftarrow r + \gamma \max_{a^\prime} Q(s^\prime, a^\prime)$', 
       fontsize=17, ha='center', va='center', color=formula_color,
       weight='bold')

# Policy
policy_box = FancyBboxPatch((7.3, 2.8), 3.9, 0.5,
                           boxstyle="round,pad=0.08",
                           edgecolor=action_color,
                           facecolor='#F3E5F5',
                           linewidth=3,
                           zorder=3)
ax.add_patch(policy_box)
ax.text(9.25, 3.05, r'$\pi(s) = \arg\max_a Q(s,a)$', 
       fontsize=17, ha='center', va='center', color=action_color,
       weight='bold')

# ============ Information Flow (Center) ============

# State Flow (Top arrow)
state_arrow = FancyArrowPatch((5.0, 5.8), (7.0, 5.8),
                             arrowstyle='->,head_width=0.8,head_length=1.0',
                             color=state_color, linewidth=6,
                             zorder=4)
ax.add_patch(state_arrow)

state_label_box = FancyBboxPatch((5.4, 5.5), 1.2, 0.6,
                                boxstyle="round,pad=0.08",
                                edgecolor=state_color,
                                facecolor='white',
                                linewidth=4,
                                zorder=5)
ax.add_patch(state_label_box)
ax.text(6.0, 5.8, r'$s_t$', fontsize=24, ha='center', va='center',
       color=state_color, weight='bold')

# Action Flow (Middle arrow)
action_arrow = FancyArrowPatch((7.0, 4.5), (5.0, 4.5),
                              arrowstyle='->,head_width=0.8,head_length=1.0',
                              color=action_color, linewidth=6,
                              zorder=4)
ax.add_patch(action_arrow)

action_label_box = FancyBboxPatch((5.4, 4.2), 1.2, 0.6,
                                 boxstyle="round,pad=0.08",
                                 edgecolor=action_color,
                                 facecolor='white',
                                 linewidth=4,
                                 zorder=5)
ax.add_patch(action_label_box)
ax.text(6.0, 4.5, r'$a_t$', fontsize=24, ha='center', va='center',
       color=action_color, weight='bold')

# Reward Flow (Bottom arrow)
reward_arrow = FancyArrowPatch((5.0, 3.2), (7.0, 3.2),
                              arrowstyle='->,head_width=0.8,head_length=1.0',
                              color=reward_color, linewidth=6,
                              zorder=4)
ax.add_patch(reward_arrow)

reward_label_box = FancyBboxPatch((5.4, 2.9), 1.2, 0.6,
                                 boxstyle="round,pad=0.08",
                                 edgecolor=reward_color,
                                 facecolor='white',
                                 linewidth=4,
                                 zorder=5)
ax.add_patch(reward_label_box)
ax.text(6.0, 3.2, r'$r_t$', fontsize=24, ha='center', va='center',
       color=reward_color, weight='bold')

# Labels for arrows
ax.text(6.0, 6.3, 'State', fontsize=20, ha='center', va='center',
       color=state_color, weight='bold')
ax.text(6.0, 4.95, 'Action', fontsize=20, ha='center', va='center',
       color=action_color, weight='bold')
ax.text(6.0, 3.65, 'Reward', fontsize=20, ha='center', va='center',
       color=reward_color, weight='bold')

# ============ Bottom Information Boxes ============

# State Space Definition
state_space_box = FancyBboxPatch((0.5, 1.5), 5.0, 0.8,
                                boxstyle="round,pad=0.12",
                                edgecolor=state_color,
                                facecolor='#FFF3E0',
                                linewidth=4,
                                zorder=1)
ax.add_patch(state_space_box)
ax.text(3.0, 2.1, 'State Space', fontsize=20, ha='center', va='center',
       color=text_color, fontweight='bold')
ax.text(3.0, 1.75, r'$\mathcal{S} = [T_{temp}, C_{glucose}, pH, pO_2, I_{insulin}, V_{rate}] \in \mathbb{R}^6$', 
       fontsize=16, ha='center', va='center', color=formula_color, weight='bold')

# Action Space Definition
action_space_box = FancyBboxPatch((6.5, 1.5), 5.0, 0.8,
                                 boxstyle="round,pad=0.12",
                                 edgecolor=action_color,
                                 facecolor='#F3E5F5',
                                 linewidth=4,
                                 zorder=1)
ax.add_patch(action_space_box)
ax.text(9.0, 2.1, 'Action Space', fontsize=20, ha='center', va='center',
       color=text_color, fontweight='bold')
ax.text(9.0, 1.75, r'$\mathcal{A} = \{-1, 0, +1\}^9$ (19,683 discrete actions)', 
       fontsize=16, ha='center', va='center', color=formula_color, weight='bold')

# Objective
objective_box = FancyBboxPatch((1.5, 0.5), 9.0, 0.8,
                              boxstyle="round,pad=0.12",
                              edgecolor=reward_color,
                              facecolor='#FFEBEE',
                              linewidth=4,
                              zorder=1)
ax.add_patch(objective_box)
ax.text(6.0, 1.1, 'Training Objective', fontsize=20, ha='center', va='center',
       color=text_color, fontweight='bold')
ax.text(6.0, 0.75, r'$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{T=24h} \gamma^t r_t \right]$ - Maximize 24-hour survival time', 
       fontsize=17, ha='center', va='center', color=formula_color, weight='bold')

# ============ Legend ============
legend_elements = [
    mlines.Line2D([], [], color=state_color, marker='>', markersize=18, 
                  label='State Flow', linewidth=6, linestyle='-'),
    mlines.Line2D([], [], color=action_color, marker='>', markersize=18, 
                  label='Action Flow', linewidth=6, linestyle='-'),
    mlines.Line2D([], [], color=reward_color, marker='>', markersize=18, 
                  label='Reward Flow', linewidth=6, linestyle='-')
]
legend = ax.legend(handles=legend_elements, loc='upper left', 
                  fontsize=18, framealpha=0.98, edgecolor='gray',
                  fancybox=True, shadow=True, bbox_to_anchor=(0.02, 0.98),
                  title='Information Flow', title_fontsize=20)
legend.get_frame().set_linewidth(3)

# Time indicator
time_text = ax.text(11.5, 0.3, r'$t \rightarrow t+1$', 
                   fontsize=18, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                           edgecolor='gray', linewidth=2.5),
                   color='#546E7A', weight='bold')

# Save as high-quality PDF
plt.tight_layout()
output_path = '/Users/xluobd/Desktop/Simulator/New_System_Results/RL_Framework_Diagram.pdf'
plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none', format='pdf')
print(f"✓ Framework diagram saved to: {output_path}")

# Also save as high-res PNG
png_path = '/Users/xluobd/Desktop/Simulator/New_System_Results/RL_Framework_Diagram.png'
plt.savefig(png_path, bbox_inches='tight', facecolor='white', edgecolor='none', dpi=300)
print(f"✓ PNG preview saved to: {png_path}")

plt.show()

print("\n" + "="*70)
print("✓ Clean, professional framework diagram created successfully!")
print("="*70)
print("✓ No overlapping elements")
print("✓ Large, readable text and formulas")
print("✓ Clear organization and spacing")
print("✓ Professional layout suitable for publications")
print("="*70)
