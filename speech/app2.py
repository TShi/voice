import matplotlib.pyplot as plt
import numpy as np
import Tkinter as tk
import matplotlib.figure as mplfig
import matplotlib.backends.backend_tkagg as tkagg
pi = np.pi
from Tkinter import *
from model import *

import matplotlib
font = {
        'size'   : 22}

matplotlib.rc('font', **font)

class App(Frame):
    def __init__( self, parent, width=500, height=100 ):
        self.parent = parent
        Frame.__init__( self, parent )
        self.fig = mplfig.Figure(figsize=(6,6))
        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.parent)
        self.ax = self.fig.add_subplot(111)
        self.probs = np.array([1./3,1./3,1./3])
        self.update()
    def update(self):
        signal = MY_REC.record()
        if not signal:
            self.ax.set_title("(Too Short)")
            self.update_idletasks()
            self.after_idle( self.update )
            return
        fund_freq,X = FeatureExtractor.get_features(MY_REC.fs,signal)
        if not X:
            self.ax.set_title("(Missed it)")
            self.update_idletasks()
            self.after_idle( self.update )
            return
        all_prob = MY_VOICE_CLF.predict_proba(X)[0]
        self.probs = self.probs * 0.8 + all_prob * 0.2
        self.ax.cla()
        self.update_idletasks()
        vals = [np.random.random(),0.3,0.4]
        self.ax.set_title(PERSONS[np.argmax(self.probs)])
        print all_prob
        bars = (self.ax.bar([0,1,2], self.probs, width=0.8))
        self.ax.set_xticks([0.4,1.4,2.4])
        self.ax.set_xticklabels(PERSONS)
        self.ax.set_yticks([0,0.3,0.6])
        self.ax.set_ylim([0,1])
        self.canvas.get_tk_widget().pack()
        self.canvas.draw()
        self.update_idletasks()
        self.after_idle( self.update )



MY_REC = Recorder(True)
MY_VOICE_MANAGER = VoiceManager(FeatureExtractor)
MY_VOICE_CLF=VoiceClassifier(clf=SVC(probability=True))
MY_VOICE_CLF.fit(MY_VOICE_MANAGER.X, MY_VOICE_MANAGER.y)
PERSONS = MY_VOICE_CLF.get_classes()
assert len(PERSONS) == 3

def main():
    root = tk.Tk()
    app = App(root)
    tk.mainloop()

if __name__ == '__main__':
    main()