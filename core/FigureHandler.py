from matplotlib import pyplot as plt


class FigureHandler:
    def __init__(self, layout="a"):
        self.layout = layout
        self.plots = plt.subplot_mosaic

    def plot(self, mosaic, *, sharex=False, sharey=False, **fig_kw):
        self.plots = plt.subplot_mosaic(mosaic=mosaic, sharex=sharex, sharey=sharey, **fig_kw)


if __name__ == "__main__":
    layout = """
    AAAC
    BBBC"""
    fh = FigureHandler(layout=layout)
    fh.plot(fh.layout)
    plt.show()