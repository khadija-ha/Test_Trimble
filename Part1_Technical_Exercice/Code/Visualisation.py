import numpy as np
import visdom
import torch

class VisualisationHandler:

    def __init__(self):

        #Initialise visual environnemeent dictionnary
        self.m_visualEnvDict=dict()

        #Initialize idDict
        self.m_idDict=dict()

        #Create main environnement visualiser and associate it with general key
        self.createEnvironnementContent("main")

        
    def createEnvironnementContent(self, nameEnv):

        #Create visual
        self.m_visualEnvDict[nameEnv]=visdom.Visdom(env=nameEnv)

    def initializePlotFigure(self, nameEnv, nameWindow):

        #Initialize figure
        self.m_idDict[nameWindow] = self.m_visualEnvDict[nameEnv].line(X=np.array([0]),Y=np.array([0]), opts=dict(title=nameWindow))
    
    def updatePlotFigure(self, nameEnv, nameWindow, x, y):

        #Add Point
        if nameWindow in self.m_idDict:
            self.m_visualEnvDict[nameEnv].line(X=np.array([x]),Y=np.array([y]),win=self.m_idDict[nameWindow], update='append', opts=dict(title=nameWindow))
        else:
            self.m_idDict[nameWindow] = self.m_visualEnvDict[nameEnv].line(X=np.array([x]),Y=np.array([y]), opts=dict(title=nameWindow))

    def updateMainCriteriaFigure(self, x, y):

        #Update main criteria figure
        self.updatePlotFigure("main", "Evolution of the main criteria", x, y)

    def updateTrainLoss_EvolutionFigure(self,x,y):
        
        #Update main criteria figure
        self.updatePlotFigure("main", "Evolution of the train loss value", x, y)

    def updateValLoss_EvolutionFigure(self,x,y):
        
        #Update main criteria figure
        self.updatePlotFigure("main", "Evolution of the val loss value", x, y)

