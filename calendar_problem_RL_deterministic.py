import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import time
from IPython.display import clear_output
from matplotlib import transforms, animation
import imageio
from IPython.display import HTML

def rainbow_text(x, y, ls, lc, ax, fig,**kw):
    t = ax.transData
    for s,c in zip(ls,lc):
        text = ax.text(x, y, s, color=c, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')
		
		
		
global state_locations, actions, state_rewards, state_q_values, state_actions, current_algorithm_step, previous_state_action, current_state_action, is_first_iteration, alpha, new_previous_q_val_memory, current_state_color, previous_state_color
def reset():
    global state_locations, actions, state_rewards, state_q_values, state_actions, current_algorithm_step, previous_state_action, current_state_action, is_first_iteration, alpha, new_previous_q_val_memory, current_state_color, previous_state_color
    state_q_values = {0: [0.1, 0],
                      1: [0, 0.1],
                      2: [0, 0.1],
                      3: [0, 0],
                      4: [0, 0.1],
                      5: [0, 0],
                      6: [0, 0],
                      7: [0, 0],
                      8: [0, 0]}
    state_actions = {0: [1, 2],
                     1: [3, 4],
                     2: [5, 6],
                     4: [7, 8]}
    state_locations = [(0.15, 0.5), 
                       (0.4, 0.775), (0.4, 0.225),
                       (0.65, 0.9), (0.65, 0.65), (0.65, 0.35), (0.65, 0.1),
                       (0.9, 0.8), (0.9, 0.525)]
    actions = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (4, 7), (4, 8)]
    state_rewards = {1: 4,
                     2: 1,
                     3: 10,
                     4: 1,
                     5: 2,
                     6: -3,
                     7: -3,
                     8: -4}
    current_algorithm_step = -1
    current_state_action = [0, np.argmax(state_q_values[0])]
    previous_state_action = None
    previous_state_color = 'C5'
    current_state_color = 'C1'
    is_first_iteration = True
    alpha = 1
    new_previous_q_val_memory = None
    return ()

def get_current_rl_step():
    if current_algorithm_step == -1:
        return "initialize Q values"
    
    if current_algorithm_step == 0:
        return "start"
    
    if current_algorithm_step == 1:
        return "action selection rule"
    
    if current_algorithm_step == 2:
        return "take action from previous state"
    
    if current_algorithm_step == 3:
        return "update Q values"
    
    if current_algorithm_step == 4:
        return "update previous/current state-actions"
    

def draw_calendar_problem(ax, fig):
    global state_q_values, state_locations, actions, state_rewards, previous_state_action, current_state_action, is_first_iteration, alpha, new_previous_q_val_memory
    ax.cla()
    ax.axis('off')
    
    state_radius = 0.09
    
    for state, state_location in enumerate(state_locations):
        edgecolor = current_state_color if state == current_state_action[0] and (current_algorithm_step >= 1 or current_state_action[0] != 0) else 'k'
        lw = 5 if state == current_state_action[0] and (current_algorithm_step >= 1 or current_state_action[0] != 0) else 1
        
        if previous_state_action is not None:
            if state == previous_state_action[0]:
                edgecolor = previous_state_color
                lw = 5
                
        circle = plt.Circle(state_location, state_radius, edgecolor=edgecolor, linewidth=lw, 
                    facecolor='white', zorder=2, alpha=1)
        ax.add_patch(circle)
        
        if state in state_actions:
            for action in state_actions[state]:
                #textcolor = current_state_color if state == current_state_action[0] and max(state_q_values[state]) ==  state_q_values[state][state_actions[state].index(action)] and current_algorithm_step >= 2 else 'k'
                textcolor = current_state_color if state == current_state_action[0] and state_actions[state].index(action) ==  current_state_action[1] and current_algorithm_step >= 2 else 'k'
                if previous_state_action is not None:
                    #if state == previous_state_action[0] and max(state_q_values[previous_state_action[0]]) ==  state_q_values[previous_state_action[0]][state_actions[previous_state_action[0]].index(action)]:
                    if state == previous_state_action[0] and state_actions[state].index(action) ==  previous_state_action[1]:
                        textcolor = previous_state_color
                if textcolor == 'k':
                    actioncolor = 'gray'
                else:
                    actioncolor = textcolor
                xs = [state_locations[state][0], state_locations[action][0]]
                ys = [state_locations[state][1], state_locations[action][1]]
                ax.plot(xs, ys, zorder=1, color=actioncolor, linewidth=10)
            
        
        if current_algorithm_step != -1:            
            offset_multiplier = 1
            for action in [0, 1]:
                textcolor = current_state_color if state == current_state_action[0] and action ==  current_state_action[1] and current_algorithm_step >= 2 else 'k'

                if previous_state_action is not None:
                    if state == previous_state_action[0] and action ==  previous_state_action[1]:
                        textcolor = previous_state_color
                
                if previous_state_action is not None and new_previous_q_val_memory is not None:
                    if state == previous_state_action[0] and action == previous_state_action[1]:
                        ax.text(state_location[0], state_location[1] + (offset_multiplier*0.035), 
                                f'{state_q_values[state][action]}→{new_previous_q_val_memory:.1f}',
                                ha='center', va='center', fontsize=10, color=textcolor, fontweight='bold')
                    else:
                        ax.text(state_location[0], state_location[1] + (offset_multiplier*0.035), 
                                f'{state_q_values[state][action]}',
                                ha='center', va='center', fontsize=15, color=textcolor)
                else:
                    ax.text(state_location[0], state_location[1] + (offset_multiplier*0.035), 
                            f'{state_q_values[state][action]}',
                            ha='center', va='center', fontsize=15, color=textcolor)
                offset_multiplier *= -1
        else:
            ax.text(state_location[0], state_location[1], f"State {state}",
                    ha='center', va='center', fontsize=15, color='k')
        
        
    for state in state_rewards:
        reward = state_rewards[state]
        state_location = state_locations[state]
        ax.text(state_location[0], state_location[1] + state_radius + 0.035, reward, color='r', 
                ha='center', va='center', fontsize=15, fontweight='bold')
            
    if current_algorithm_step >= 0:
        ax.text(state_locations[0][0] - 0.01, state_locations[0][1] + 0.125, "start", color='k',
                fontsize=15, fontweight='bold', ha='center', va='center')

    if current_algorithm_step == -1:
        ax.annotate("state", xy=(state_locations[0][0],
                                 state_locations[0][1] + state_radius + 0.01),
                    xytext=[0.1, 0.75], 
                    arrowprops=dict(facecolor='black', shrink=0),
                    ha='right', va='center', fontsize=15)
        ax.annotate("action", xy=(state_locations[0][0] + state_radius,
                                  state_locations[0][1] + state_radius + state_radius - 0.01),
                    xytext=[0.1, 0.85], 
                    arrowprops=dict(facecolor='black', shrink=0),
                    ha='right', va='center', fontsize=15)
        ax.annotate("reward", xy=(state_locations[1][0] - 0.025,
                                  state_locations[1][1] + state_radius + 0.04),
                    xytext=[0.15, 0.95], 
                    arrowprops=dict(facecolor='black', shrink=0),
                    ha='right', va='center', fontsize=15)
        ax.annotate("terminal state", xy=(state_locations[8][0],
                    state_locations[8][1] - state_radius - 0.02),
                    xytext=[1.05, 0.25], 
                    arrowprops=dict(facecolor='black', shrink=0),
                    ha='right', va='center', fontsize=15)
        
        
def draw_current_step(ax, fig):
    global is_first_iteration, current_state_action, previous_state_action, alpha, new_previous_q_val_memory
    
    ax.cla()
    ax.axis('off')
    
    ax.set_title('Next Step:', fontsize=20, y=0.8)
    
    current_step = get_current_rl_step()
    
    if current_step == "initialize Q values":
        current_step_text = "assign initial\nQ values to\neach state-action pair\n\nterminal states are assigned\nQ values of 0"
        ax.text(0.5, 0.4, current_step_text, ha='center', va='center', 
                fontsize=20, color='k')
    
    if current_step == "start":
        if is_first_iteration:
            current_step_text = "begin at\nthe start state"
        else:
            current_step_text = 'restart at the\nstart state because\ntermimal state\nreached and set the\n"previous state-action"\nto None'
            current_state_action = [0, np.argmax(state_q_values[0])]
            previous_state_action = None
            is_first_iteration = True
            ax.set_title('Next Step:', fontsize=20, y=0.9)
            
        ax.text(0.5, 0.5, current_step_text, ha='center', va='center', 
                fontsize=20, color='k')
        
    if current_step == "action selection rule":
        
        if not is_first_iteration:
            ax.text(0.5, 0.75, "action selection rule", fontsize=20, color='k', va='center', ha='center')

            current_step_text = 'choose action in the\ncurrent state with\nmaximal Q value'
            ax.text(0.5, 0.45, current_step_text, ha='center', va='center', 
                    fontsize=20, color='k')
            current_step_text = '(if no valid actions,\n go back to start state)'
            ax.text(0.5, 0.1, current_step_text, ha='center', va='center', 
                    fontsize=15, color='k')
            ax.set_title('Next Step:', fontsize=20, y=0.9)
        else:
            ax.text(0.5, 0.65, "action selection rule", fontsize=20, color='k', va='center', ha='center')

            current_step_text = 'choose action in the\ncurrent state with\nmaximal Q value'
            ax.text(0.5, 0.35, current_step_text, ha='center', va='center', 
                    fontsize=20, color='k')

        
    if current_step == "take action from previous state":
        ax.text(0.5, 0.8, f"current state-action\n Q(s', a') = " + f'{state_q_values[current_state_action[0]][current_state_action[1]]}', 
                ha='center', va='center', 
                fontsize=20, color=current_state_color)
        
        current_state_reward = state_rewards[current_state_action[0]] if current_state_action[0] in state_rewards else 0
        ax.text(0.5, 0.55, f"current state-reward\n R(s') = " + f'{current_state_reward:.2f}', 
                ha='center', va='center', 
                fontsize=20, color=current_state_color)
        
        if previous_state_action is not None:
            ax.text(0.5, 0.25, f"previous state-action\n Q(s', a') = " + f'{state_q_values[previous_state_action[0]][previous_state_action[1]]}', 
                    ha='center', va='center', 
                    fontsize=20, color=previous_state_color)
        else:
            ax.text(0.5, 0.15, f"previous state-action\n Q(s, a) = None", 
                ha='center', va='center', 
                fontsize=20, color=previous_state_color)
            
        ax.set_title('')
            
    if current_step == "update previous/current state-actions":
        if is_first_iteration:
            current_step_text = 'since presently no\n"previous state-action" exists,\nskip Q-value update step'
            ax.text(0.5, 0.75, current_step_text, ha='center', va='center', 
                    fontsize=15, color='k')
            ax.text(0.5, 0.25, 'follow maximal Q value action\nfrom current state to\nget the next "current state",\nand set the next\n"previous state-action"\nto current state-action', ha='center', va='center', 
                    fontsize=15, color='k')
            is_first_iteration = False
            ax.set_title('Next Step:', fontsize=20, y=1)
        else:
            ax.text(0.5, 0.8, 'update previous state-action Q value', ha='center', va='center', 
                    fontsize=18, color='k')
            ax.text(0.5, 0.3, 'follow maximal Q value action\nfrom current state to\nget the next "current state",\nand set the next\n"previous state-action"\nto current state-action', ha='center', va='center', 
                    fontsize=18, color='k')
            ax.set_title('Next Step:', fontsize=20, y=1)
        
        if new_previous_q_val_memory is not None:
            state_q_values[previous_state_action[0]][previous_state_action[1]] = new_previous_q_val_memory
            new_previous_q_val_memory = None

    
    if current_step == "update Q values":
        current_step_text = "update previous state-action's\nQ value"
        ax.text(0.5, 0.8, current_step_text, 
                ha='center', va='center', fontsize=20, color='k')
        #ax.text(0.5, 0.5, "prediction error =\nR(s') + Q(s', a') - Q(s, a)",
        #        ha='center', va='center', fontsize=15, color='k')
        rainbow_text(-0.1, 0.55, 
                ls=["prediction error", " = ", "R(s')", " + ", "Q(s', a')", " - ", "Q(s, a)"],
                lc=['C0', 'k', 'red', 'k', current_state_color, 'k', previous_state_color],
                ha='left', va='center', fontsize=15, ax=ax, fig=fig, fontweight='bold')
        
        r_s = 0 if current_state_action[0] not in state_rewards else state_rewards[current_state_action[0]]
        current_q_val = state_q_values[current_state_action[0]][current_state_action[1]]
        previous_q_val = state_q_values[previous_state_action[0]][previous_state_action[1]]
        pred_err = r_s + current_q_val - previous_q_val
        ax.text(0.5, 0.4, f"{pred_err:.2f} = {r_s:.2f} + {current_q_val:.2f} - {previous_q_val:.2f}",
               ha='center', va='center', fontsize=20, color='k')
        
        #ax.text(0.5, 0.2, r"Q(s, a) = Q(s, a) + ($\alpha$ * prediction error)",
        #        ha='center', va='center', fontsize=15, color='k')
        rainbow_text(-0.1, 0.15, 
                ls=["Q(s, a)", " = ", "Q(s, a)", " + (", r"$\alpha$", " * ", "prediction error", ")"],
                lc=[previous_state_color, 'k', previous_state_color, 'k', 'C0', 'k', 'C0', 'k'],
                ha='left', va='center', fontsize=15, ax=ax, fig=fig, fontweight='bold')
        
        new_previous_q_val = previous_q_val + (alpha * pred_err)
        ax.text(0.5, 0.0, f"{new_previous_q_val:.2f} = {previous_q_val:.2f} + ({alpha:.2f} * {pred_err:.2f})",
                ha='center', va='center', fontsize=20, color='k')
        new_previous_q_val_memory = round(new_previous_q_val, 2)
        
        ax.set_title('Next Step:', fontsize=20, y=1)
        
        
def update_algorithm():
    global current_algorithm_step, previous_state_action, current_state_action, is_first_iteration
    
    if current_algorithm_step == 2 and previous_state_action is None:
        current_algorithm_step += 2
    else:
        current_algorithm_step += 1
        
    if current_algorithm_step == 5:
        if previous_state_action is None:
            previous_state_action = current_state_action
            current_state = state_actions[previous_state_action[0]][previous_state_action[1]]
            current_action = np.argmax(state_q_values[current_state])
            current_state_action = [current_state, current_action]
        elif current_state_action[0] in state_actions:
            previous_state_action = current_state_action
            current_state = state_actions[previous_state_action[0]][previous_state_action[1]]
            current_action = np.argmax(state_q_values[current_state])
            current_state_action = [current_state, current_action]
        else:
            current_algorithm_step = 0         
    
    if current_algorithm_step >= 5:
        current_algorithm_step = 1
