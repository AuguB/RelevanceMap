
import math

from scipy.stats import beta
import numpy as np
import cairo

# We make a class that takes care of the whole image generation procedure
class Blobmaker:
    
    def sample_l(self, n = 1):
        # We sample from a beta distribution with a slight bias towards 0.5
        peakiness = 2
        sample = beta.rvs(peakiness, peakiness, size=n)
        return sample

    def reparam_beta(self, m, s):
        # we reduce s to satisfy s < m(1-m) if necessary
        s = np.where(s >= m*(1-m), m*(1-m)-1e-10, s)
        frac = m*(1-m)/s - 1
        a = m*frac
        b = (1-m)*frac
        return a, b

    def sample_widths_and_colors(self, l):
        # We sample the width and color values again from a beta distribution.
        # We use a reparameterization to center the betas at l.
        width_var = color_var = 0.01
        width_alphas, width_betas = self.reparam_beta(l, width_var)
        color_alphas, color_betas = self.reparam_beta(l, color_var)
        widths = beta.rvs(a=np.tile(width_alphas,(4,1)).T,
                        b=np.tile(width_betas,(4,1)).T, scale=0.3)
        # We use RGB colors
        colors = beta.rvs(a=np.tile(color_alphas,(3,4,1)).T,
                        b=np.tile(color_betas,(3,4,1)).T)
        return widths, colors

    def make_blob_image(self, widths, colors, imwidth=33, imheight=33):
        xs = [0.25, 0.75, 0.25, 0.75]
        ys = [0.25, 0.25, 0.75, 0.75]
        with cairo.ImageSurface(cairo.FORMAT_ARGB32, imwidth, imheight) as surface:
            context = cairo.Context(surface)
            context.scale(imwidth, imwidth)
            context.set_source_rgb(0.5, 0.5, 0.5)
            context.paint()
            for j, (x, y) in enumerate(zip(xs, ys)):
                context.move_to(x, y)
                context.arc(x, y, widths[j], 0, 2*math.pi)
                context.close_path()
                context.set_source_rgb(*colors[j])
                context.fill()
            buf = surface.get_data()
            data = np.ndarray(shape=(imwidth, imheight, 4),
                                dtype=np.uint8,
                                buffer=buf)
            return data.astype(np.uint8)[:,:,:3]

    def make_blob_images(self, widths, colors, imwidth=33, imheight=33):
        n_images = widths.shape[0]
        blobs_data = np.empty((n_images, imwidth, imheight, 3), dtype=int)
        for i, (width, color) in enumerate(zip(widths, colors)):
            blob_image = self.make_blob_image(width, color, imwidth, imheight)
            blobs_data[i] = blob_image
        blobs_data = np.swapaxes(blobs_data, 1, 3).astype(float)/255.
        return blobs_data