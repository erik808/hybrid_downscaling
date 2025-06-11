import numpy as np
from matplotlib import pyplot as plt

class TransectPicker:
    '''

    Create a transect. Select points, then straight paths are drawn in
    between. After completion a TransectPicker object will contain
    x,y-indices in the public x_trans and y_trans members.

        Usage:     left mouse button  - select points
                   right mouse button - undo
                   backspace          - reset
                   enter              - exit

    '''

    def __init__(self, image, data):
        self.__welcome_message()

        self.x = []
        self.y = []

        self.__image = image

        clmap = self.__image.get_cmap().copy()
        clmap.set_over('r')

        self.__im_trans = np.ones(data.shape)*np.nan

        vmin = np.min(image.get_array())
        vmax = np.max(image.get_array())
        self.__vmax = vmax
        self.__path = plt.pcolormesh(self.__im_trans,
                                     cmap = clmap,
                                     vmin = vmin,
                                     vmax = vmax)

        self.x_sections = []
        self.y_sections = []

        self.__cid_but = image.figure.canvas.mpl_connect('button_press_event', self)
        self.__cid_key = image.figure.canvas.mpl_connect('key_press_event', self)
        self.__initialize = True

    def __call__(self, event):

        if event.name == 'key_press_event':

            if event.key == 'enter':
                self.__create_transect()
                self.__exit()
            elif event.key == 'backspace':
                self.__reset()

        elif event.name == 'button_press_event':
            if event.inaxes is None:
                print('cursor not in axes')
                return

            # append to list of nodes
            if (event.button == 1):
                self.__append(event.xdata, event.ydata)

            # undo on right button click
            elif (event.button == 3):
                self.__undo()

    def __create_transect(self):

        if (len(self.x) > 1) and (len(self.y) > 1):

            self.x_trans = int(self.x[0])
            self.y_trans = int(self.y[0])

            for i in np.arange(len(self.x_sections)):
                xinds = self.x_sections[i][0]
                yinds = self.y_sections[i][0]

                self.x_trans = np.append(self.x_trans, xinds[1:])
                self.y_trans = np.append(self.y_trans, yinds[1:])

        else:
            self.x_trans = []
            self.y_trans = []


    def __create_section(self):
        '''
        Create a set of grid indices that describe a straight path
        from p0 to p1.

        '''
        if (len(self.x) < 2) and (len(self.y) < 2):
            return

        p1 = [self.x[-1], self.y[-1]]
        p0 = [self.x[-2], self.y[-2]]

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        # length = (maximum distance in the grid) + 1
        length = int(np.maximum(np.abs(dx), np.abs(dy)) + 1)

        xinds = np.linspace(p0[0], p1[0], length).astype(int)
        yinds = np.linspace(p0[1], p1[1], length).astype(int)

        self.x_sections.append([xinds])
        self.y_sections.append([yinds])

        return xinds, yinds

    def __welcome_message(self):
        print("Welcome to the transect picker")
        print("Usage:     left mouse button  - select points")
        print("           right mouse button - undo ")
        print("           backspace          - reset")
        print("           enter              - exit")

    def __append(self, xdata, ydata):

        # map x,y-data to cell centers
        self.x.append(np.floor(xdata) + 0.5)
        self.y.append(np.floor(ydata) + 0.5)
        self.__create_section()
        self.__draw()

    def __reset(self):
        if (len(self.x) > 0) and (len(self.y) > 0):
            print('reset')
            self.x = []
            self.y = []
            self.x_sections = []
            self.y_sections = []
            self.__draw()
        else:
            print('nothing to reset')

    def __undo(self):
        if (len(self.x) > 0):
            print('undo')
            self.x.pop()
            self.y.pop()

            if (len(self.x_sections) > 0):
                print('popping xsections')
                self.x_sections.pop()
                self.y_sections.pop()

            self.__draw()
        else:
            print('nothing to undo')

    def __draw(self):
        self.__print()

        if self.__initialize:
            self.__line, = self.__image.axes.plot(self.x, self.y, 'k.-')
            self.__initialize = False
        else:
            self.__line.set_data(self.x, self.y)

        self.__create_transect()
        self.__im_trans[:] = np.nan
        self.__im_trans[self.y_trans, self.x_trans] = self.__vmax + 1
        self.__path.set_array(self.__im_trans)
        self.__image.figure.canvas.draw_idle()

    def __print(self):
        print('points:')
        lst = np.floor([self.x, self.y]).astype(int)
        for item in zip(*lst):
            print('{0:4d} {1:4d}'.format(*item))

    def __exit(self):
        print('exiting')
        self.__image.figure.canvas.mpl_disconnect(self.__cid_but)
        self.__image.figure.canvas.mpl_disconnect(self.__cid_key)
        plt.close(self.__image.figure)
