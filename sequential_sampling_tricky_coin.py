import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
from IPython.display import clear_output
import matplotlib
from tqdm.notebook import tqdm
import sys
from base64 import b64encode

matplotlib.rcParams['animation.embed_limit'] = 2**256

def chooseCoin():
    rule = np.random.choice(["Trick","Fair"], p=[0.5, 0.5])
    return rule


def flipCoin(coin):
    if coin=="Trick":
        probs = [0.6, 0.4]
    else:
        probs = [0.5, 0.5]
    side = np.random.choice(["heads", "tails"], p=probs)
    return side


def initChoice():
    global startDV, DVhistory, thresholdLow,thresholdHigh, correctBoundary, ax
    global boundaryCrossed
    global activeCoin, likelihoodsEven, likelihoodsOdd
    global threshHighLine, threshLowLine,dvLine, threshCorrectLine, threshIncorrectLine
    global simulation_length, answer_counts

    answer_counts = (0, 0, 0)  # number of sims that converge to (fair, tricky, need more data)

    ax.set_xlim((0, simulation_length))
    ax.set_ylim((-4.0, 4.0))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_ylabel('Evidence Accumulated', fontsize=25)
    ax.set_xlabel('Time', fontsize=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    threshHighLine, = ax.plot([], [], 'k--', linewidth=2) # black dash lines
    threshLowLine, = ax.plot([], [], 'k--', linewidth=2)  # for thresholds
    
    #thresholdHigh = np.log((1-alpha)/alpha)
    #thresholdLow = np.log(beta/(1-beta))
    startDV = (thresholdHigh + thresholdLow)/2
        
    init_lines()
    activeCoin = chooseCoin()
    if activeCoin == "Trick":
        correctBoundary = "high"
        threshHighLine.set_color('g')
        threshLowLine.set_color('r')
    else:
        correctBoundary = "low"
        threshHighLine.set_color('r')
        threshLowLine.set_color('g')
        
    threshHighLine.set_data([0, simulation_length], [thresholdHigh, thresholdHigh])
    threshLowLine.set_data([0, simulation_length], [thresholdLow, thresholdLow])

    return (dvLine, threshHighLine, threshLowLine,)
  
def sampleEvidence():
    global activeCoin
    return(flipCoin(activeCoin))


def updateDV(oldDV, samp):
    global p_heads_trick_coin, p_tails_trick_coin
    if samp == "heads":
        w = np.log(p_heads_trick_coin/0.5)
    else:
        w = np.log(p_tails_trick_coin/0.5)
    newDV = oldDV + w
    return(newDV)


def animateDV(t_local):
    global DVhistory, boundaryCrossed, t, pause_count, n_frames, ax
    global thresholdLow, thresholdHigh, correctBoundary
    global threshHighLine, threshLowLine, dvLine, threshCorrectLine, threshIncorrectLine
    global simulation_length, answer_counts
    
    if (t == 0) and ((n_frames - t_local) < simulation_length):
        return ()
    
    if boundaryCrossed or (t == simulation_length):
        pause_count += 1
        if pause_count > 50:
            init_lines()
        return ()
 
    if ((t == simulation_length-1) and (not boundaryCrossed)):
        if correctBoundary == "high":
            dvLine.set_color('k')
        else:
            dvLine.set_color('k')
        answer_counts = (answer_counts[0], answer_counts[1], answer_counts[2]+1)
        t += 1
        pause_count += 1
        return ()        
    
    ## Get a new noisy sample
    if (not boundaryCrossed):
        dvLine.set_data(DVhistory[:,0], DVhistory[:,1])
        samp = sampleEvidence()
        newDV = updateDV(DVhistory[t, 1], samp)
        DVhistory[t+1, :] = [t+1, newDV]
        
        if (newDV >= thresholdHigh):
            if correctBoundary == "high":
                dvLine.set_color('g')
            else:
                dvLine.set_color('r')
            boundaryCrossed = True
            answer_counts = (answer_counts[0], answer_counts[1] + 1, answer_counts[2])
            ax.set_title(f'Decision Target: {activeCoin}\nDecision Counts: Fair={answer_counts[0]}  Trick={answer_counts[1]}  Need More Data={answer_counts[2]}',
                        fontsize=20)
            t += 1
            pause_count += 1
            return ()
    
        if (newDV <= thresholdLow):
            if correctBoundary == "low":
                dvLine.set_color('g')
            else:
                dvLine.set_color('r')
            boundaryCrossed = True
            answer_counts = (answer_counts[0] + 1, answer_counts[1], answer_counts[2])
            ax.set_title(f'Decision Target: {activeCoin}\nDecision Counts: Fair={answer_counts[0]}  Trick={answer_counts[1]}  Need More Data={answer_counts[2]}',
                        fontsize=20)
            t += 1
            pause_count += 1
            return ()
            
    ax.set_title(f'Decision Target: {activeCoin}\nDecision Counts: Fair={answer_counts[0]}  Trick={answer_counts[1]}  Need More Data={answer_counts[2]}',
                 fontsize=20)
    t += 1
    return ()
  
  
  
def init_lines():
    global dvLine, threshHighLine, threshLowLine, threshCorrectLine, threshIncorrectLine, DVhistory, boundaryCrossed, t, pause_count
    if dvLine is not None:
        dvLine.set_alpha(0.1)
    dvLine, = ax.plot([], [], 'b-', linewidth=1.5)
    
    dvLine.set_data([[0]],[[startDV]])
     
    DVhistory = np.zeros((simulation_length, 2))
    DVhistory[:] = np.nan
    DVhistory[0, :] = [0, startDV]
    
    boundaryCrossed = False
    t = 0
    pause_count = 0
    
    
def run_simulation():
    clear_output(wait=True)
    global simulation_length, thresholdHigh, thresholdLow, p_heads_trick_coin, p_heads_trick_coin, ax, dvLine
    global n_frames, p_tails_trick_coin
    simulation_length = 200

    thresholdHigh = -np.inf
    thresholdLow = np.inf
    p_heads_trick_coin = -np.inf
    dvLine = None


    while thresholdLow >= thresholdHigh:
        thresholdHigh = float(input('\nEnter the upper threshold value.\nMust be between -4 and 4.\nA reasonable value is 3.\n   Your input: '))
        while (thresholdHigh < -4) or (thresholdHigh > 4):
            thresholdHigh = float(input('\nUpper threshold value must be between -4 and 4.\nEnter the upper threshold value.\nMust be between -4 and 4.\nA reasonable value is 3.\n   Your input: '))
        thresholdLow = float(input('\nEnter the lower threshold value.\nMust be between -4 and 4.\nA reasonable value is -1.5.\n   Your input: '))
        while (thresholdLow < -4) or (thresholdLow > 4):
            thresholdLow = float(input('\nLower threshold value must be between -4 and 4.\nEnter the lower threshold value.\nMust be between -4 and 4.\nA reasonable value is -1.5.\n   Your input: '))
        if thresholdLow >= thresholdHigh:
            print('\n>>> Upper threshold must be higher than lower threshold. <<<\n>>> Please re-enter these values <<<')


    implied_alpha = 1/(1+np.exp(thresholdHigh))
    implied_beta = (np.exp(thresholdLow)) / (1+np.exp(thresholdLow))

    print(f'\n   The implied type-1 error for these thresholds is {implied_alpha*100:.2f}%')
    print(f'   The implied type-2 error for these thresholds is {implied_beta*100:.2f}%')

    while (p_heads_trick_coin <= 0) or (p_heads_trick_coin >= 1):
        p_heads_trick_coin = float(input('\nEnter the probability of flipping heads with the tricky coin.\nThis is the drift rate - a value far from 0.5 yields a high drift rate.\nThis probability must be between 0 and 1.\n   Your input: ')) 
        p_tails_trick_coin = 1-p_heads_trick_coin

    def chooseCoin():
        return "Trick"

    n_frames = 2000
    fig, ax = plt.subplots(figsize=(10, 7))
    anim = animation.FuncAnimation(fig, animateDV, init_func=initChoice,
                                   frames=tqdm(range(n_frames), file=sys.stdout, desc='generating output...', initial=1, position=0), 
                                   interval=10, blit=True)
    anim.save('./trick_coin.mp4', writer="ffmpeg")
    plt.close()
