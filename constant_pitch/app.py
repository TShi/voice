
from __future__ import division
from Tkinter import *
from random import randint,random
from time import sleep


from model import *



class VoiceDemo( Frame ):
    def __init__( self, parent, width=500, height=100 ):
        self.parent = parent
        Frame.__init__( self, parent )
        self.columnconfigure( 0, weight=1 )                                     # Forces the canv object to resize any time this widget is resized 
        self.rowconfigure( 0, weight=1 )
        self.statusMessage = 'Normal'
        self.w = 0
        self.h = 0
        self.canv = Canvas(self, width=width, height=height)                                # This canvas will display the progress bar and accompanying percentage text
        self.canv.grid( row=1, column=0, sticky=N+S+E+W )
        self.canv.bind( '<Configure>', lambda e:
                        self.resize( e.width, e.height ) )
        
    
    
    def resize( self, w, h ):
        """
        Handles resize events for the canv widget.  Adjusts the height and width
        of the canvas for the progress bar calculations.
        
        Arguments:
          w: The new width
          h: The new height
        """
        self.w = w
        self.h = h
        self.canv.delete( 'frame' )
        self.canv.create_rectangle( 1, 1, self.w, self.h, outline='black',
                                    fill='gray75', tag='frame' )

    def reset( self ):
        self.canv.delete( 'bar' )
        self.canv.delete( 'text' )

        
    def drawBar( self , pct, text ):
        x0 = 2                                                                  # The bar is inset by 2 pixels
        x1 = pct * ( self.w - 3 ) + 2
        y0 = 2
        y1 = self.h
        self.canv.delete( 'bar' )
        self.canv.create_rectangle( x0, y0, x1, y1, fill='SteelBlue3',
                                    outline='', tag='bar' )
        # pctTxt = '%02.2f%%' % ( pct*100, )
        self.update_text(text)

        
    def startGen( self ):
        self.update_text("Loading Model")
        self.update_idletasks()
        self.recorder = Recorder(True)
        self.voice_manager = VoiceManager(FeatureExtractor)
        self.voice_clf=VoiceClassifier(clf=SVC(probability=True))
        self.voice_clf.fit(self.voice_manager.X, self.voice_manager.y)
        self.update_text("Ready")
        self.update_idletasks()
        self.after_idle( self.iterGen )
    def update_text(self,text):
        self.canv.delete( 'text' )
        self.canv.create_text( self.w/2, self.h/2, text=text,
                               anchor=CENTER, tag='text' ,font=("Purisa",40))
    def iterGen( self ):
        signal = self.recorder.record()
        if signal:
            fund_freq,X = FeatureExtractor.get_features(self.recorder.fs,signal)
            if X:
                pred,prob = map(lambda x:x[0],self.voice_clf.predict(X))
                self.drawBar(prob, pred)
            else:
                self.drawBar(0,"Unknown")
        else:
            self.drawBar(0,"Too short...")
            
        self.update_idletasks()
        self.after_idle( self.iterGen )
        

def main():
    root = Tk()
    root.title( 'Constant Pitch Demo' )
    pgress = VoiceDemo( root )
    pgress.grid( row=1 )
    root.after(100, pgress.startGen)
    root.mainloop()

if __name__=='__main__':
    main()
