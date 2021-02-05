import torchvision.utils as vutils
import neptune
import matplotlib.pyplot as plt

def save_image(x, name, it, filename, normalize=True, x_range=(0., 1.)):
    vutils.save_image(x, filename, nrow = 10, normalize=normalize, range=x_range)
    neptune.send_image(name, x=it, y=filename)

def save_scatter(x, y, filedir, name, it, other_extensions = ['pdf']):
    filename = '{}/{}_epoch_{}.'.format(filedir, name, it+1)
    extensions = ['png'] + other_extensions
    plt.scatter(x, y, s=1)
    for ext in extensions:
        plt.savefig(filename+'{}'.format(ext))
    plt.close()
    neptune.send_image(name, x=it, y=filename+'png')
    

def get_gin_params_as_dict(gin_config):
    params = {}
    for k, v in gin_config.items():
        for (kk, vv) in v.items():
            param_name = '.'.join(filter(None, k)) + '.' + kk
            param_value = vv
            params[param_name] = param_value

    return params
