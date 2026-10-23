import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of HxW or HxWxC images."""
        for i, img in enumerate(images):
            img = np.asarray(img)
            dataformats = 'HW' if img.ndim == 2 else 'HWC'
            self.writer.add_image('%s/%d' % (tag, i), img, step,
                                  dataformats=dataformats)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(tag, values, step, bins=bins)
        self.writer.flush()
