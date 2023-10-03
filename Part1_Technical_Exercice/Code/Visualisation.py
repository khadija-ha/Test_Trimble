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

    def updateTrain_LossEvolutionFigure(self,x,y):
        
        #Update main criteria figure
        self.updatePlotFigure("main", "Evolution of the train loss value", x, y)

    def updateValLoss_EvolutionFigure(self,x,y):
        
        #Update main criteria figure
        self.updatePlotFigure("main", "Evolution of the val loss value", x, y)



    def addImagesFigure(self, nameEnv, nameWindow, titleShown, imagesLists, paddingUsed, rgbBackground):
        # Determine the maximum dimensions for all images
        max_height = max(image.shape[0] for image in imagesLists)
        total_width = sum(image.shape[1] for image in imagesLists)

        # Resize images to have consistent dimensions
        resized_images = []
        for image in imagesLists:
            h, w, c = image.shape
            scale_factor = max_height / h
            new_width = int(w * scale_factor)
            resized_image = torch.nn.functional.interpolate(
                torch.tensor(image).permute(2, 0, 1).unsqueeze(0),
                size=(max_height, new_width),
                mode='bilinear',
                align_corners=False
            )
            # Transpose the dimensions here to match (3, 159, 318)
            resized_image = resized_image.squeeze().permute(0, 2, 3, 1).numpy()
            resized_images.append(resized_image)

        # Create an empty canvas for the combined image
        combined_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        current_width = 0

        for resized_image in resized_images:
            h, w, _ = resized_image.shape

            # Add the resized image to the combined image
            combined_image[:h, current_width:current_width + w] = resized_image

            # Update the current width position
            current_width += w

        # Create a Visdom window and display the combined image
        if nameWindow not in self.m_idDict:
            self.m_visualEnvDict[nameEnv].image(
                np.transpose(combined_image, (2, 0, 1)),
                win=nameWindow,
                opts=dict(title=titleShown)
            )
        else:
            self.m_visualEnvDict[nameEnv].image(
                np.transpose(combined_image, (2, 0, 1)),
                win=nameWindow,
                update='replace',
                opts=dict(title=titleShown)
            )
