# from visdom import Visdom

class Visualizer:
    def __init__(self, visualize_ena):
        self.visual_window = None
        # if visualize_ena:
        #     self.visual_window = Visdom()
        # else:
        #     self.visual_window = None

    def line(self, data, index, win, update):
        if self.visual_window is None:
            return
        else:
            return self.visual_window.line(data, index, win=win, update=update)
