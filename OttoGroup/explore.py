import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import rcParams
import matplotlib.cm as cm

dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
sequence = ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5',
            'Class_6', 'Class_7', 'Class_8', 'Class_9']

df = pd.read_csv(r'train.csv', header = 0)
df1 = df[df.target == 'Class_1']
df2 = df[df.target == 'Class_2']
df3 = df[df.target == 'Class_3']
df4 = df[df.target == 'Class_4']
df5 = df[df.target == 'Class_5']
df6 = df[df.target == 'Class_6']
df7 = df[df.target == 'Class_7']
df8 = df[df.target == 'Class_8']
df9 = df[df.target == 'Class_9']
df1 = df1.drop(['target', 'id'], axis=1).values
df2 = df2.drop(['target', 'id'], axis=1).values
df3 = df3.drop(['target', 'id'], axis=1).values
df4 = df4.drop(['target', 'id'], axis=1).values
df5 = df5.drop(['target', 'id'], axis=1).values
df6 = df6.drop(['target', 'id'], axis=1).values
df7 = df7.drop(['target', 'id'], axis=1).values
df8 = df8.drop(['target', 'id'], axis=1).values
df9 = df9.drop(['target', 'id'], axis=1).values

means = []

for c in sequence:
    sub = []
    for i in range(1, 94):
        feat = 'feat_' + str(i)
        sub.append(df[df.target == c][feat].mean())
    sub = np.array(sub)
    means.append(sub)

means = np.array(means)
for i in means:
    plt.figure()
    plt.plot(i)
    plt.grid(False)
    plt.grid(axis = 'y', color ='white', linestyle='-')

