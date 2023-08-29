import numpy as np

from gammapy.datasets import MapDataset, Datasets
from gammapy.modeling import Covariance, Fit

# Need to patch this method of the Covariance class
def _expand_factor_matrix(matrix, parameters):
    """Expand covariance matrix with zeros for frozen parameters"""
    npars = len(parameters)
    matrix_expanded = np.zeros((npars, npars))
    mask_frozen = np.array([par.frozen for par in parameters])

    mask_not_unique=[]
    for i,x in enumerate(list(parameters)):
        if list(parameters).index(x) == i:
            mask_not_unique.append(False)
        else:
            mask_not_unique.append(True)

    mask = mask_frozen | np.array(mask_not_unique)

    free_parameters = ~(mask | mask[:, np.newaxis])
    matrix_expanded[free_parameters] = matrix.ravel()
    return matrix_expanded

Covariance._expand_factor_matrix = staticmethod(_expand_factor_matrix)

class Fit_wp(Fit):
    """Fit class.

    The same Fit class as in gammapy only the __init__ method needs to be cloned so PriorDatasets are kept.

    Parameters
    ----------
    datasets : `Datasets`
        Datasets (need to clone here to keep PriorDatasets)
    """

    def __init__(self, datasets, store_trace=False):
       

        self.store_trace = store_trace
        self.datasets = datasets


# Create some special MapDatasets that offer the option to add priors to the likelihood

class PriorDatasets(Datasets):
    """Used for the KM3NeT data set. Enables prior term on the spectral parameters of the source model."""

    def __init__(self,datasets=None, nuisance=None, **kwargs):
        super().__init__(datasets, **kwargs)
        self._nuisance = nuisance

    @property
    def nuisance(self):
        return self._nuisance

    @nuisance.setter
    def nuisance(self, nuisance):
        self._nuisance = nuisance

    def stat_sum(self):
        wstat = super().stat_sum()
        liketotal = wstat
        if self.nuisance:
            liketotal += self.nuisance_alpha() + self.nuisance_beta() + self.nuisance_ecut() + self.nuisance_amp()
        return liketotal

    def stat_sum_no_prior(self):
        return super().stat_sum()

    def nuisance_alpha(self):
        alpha = self.models['nu_PD'].parameters['alpha'].value
        alpha0 = self.nuisance['alpha']
        error = self.nuisance['alpha_err']
        return (alpha-alpha0)**2 / error**2

    def nuisance_beta(self):
        beta = self.models['nu_PD'].parameters['beta'].value
        beta0 = self.nuisance['beta']
        error = self.nuisance['beta_err']
        return (beta-beta0)**2 / error**2

    def nuisance_ecut(self):
        unit = self.nuisance['ecut_unit']
        ecut = self.models['nu_PD'].parameters['e_cutoff'].quantity.to_value(unit)
        ecut0 = self.nuisance['ecut']
        error = self.nuisance['ecut_err']
        return (ecut-ecut0)**2 / error**2

    def nuisance_amp(self):
        unit = self.nuisance['amp_unit']
        amp = self.models['nu_PD'].parameters['amplitude'].quantity.to_value(unit)
        amp0 = self.nuisance['amp']
        error = self.nuisance['amp_err']
        return (amp-amp0)**2 / error**2


class PriorMapDataset(MapDataset):
    """Used for the KM3NeT data set. Enables prior term on the spectral parameters of the source model."""

    def __init__(self, nuisance=None, **kwargs):
        super().__init__(**kwargs)
        self._nuisance = nuisance

    @property
    def nuisance(self):
        return self._nuisance

    @nuisance.setter
    def nuisance(self, nuisance):
        self._nuisance = nuisance

    def stat_sum(self):
        wstat = super().stat_sum()
        liketotal = wstat
        if self.nuisance:
            liketotal += self.nuisance_alpha() + self.nuisance_beta() + self.nuisance_ecut() + self.nuisance_amp()
        return liketotal

    def stat_sum_no_prior(self):
        return super().stat_sum()

    def nuisance_alpha(self):
        alpha = self.models.parameters['alpha'].value
        alpha0 = self.nuisance['alpha']
        error = self.nuisance['alpha_err']
        return (alpha-alpha0)**2 / error**2

    def nuisance_beta(self):
        beta = self.models.parameters['beta'].value
        beta0 = self.nuisance['beta']
        error = self.nuisance['beta_err']
        return (beta-beta0)**2 / error**2

    def nuisance_ecut(self):
        unit = self.nuisance['ecut_unit']
        ecut = self.models.parameters['e_cutoff'].quantity.to_value(unit)
        ecut0 = self.nuisance['ecut']
        error = self.nuisance['ecut_err']
        return (ecut-ecut0)**2 / error**2

    def nuisance_amp(self):
        unit = self.nuisance['amp_unit']
        amp = self.models.parameters['amplitude'].quantity.to_value(unit)
        amp0 = self.nuisance['amp']
        error = self.nuisance['amp_err']
        return (amp-amp0)**2 / error**2
    
class PriorMapDataset2(MapDataset):
    """Used for the CTA data set. Enables prior term on the hadronic contribution."""

    def __init__(self, nuisance=None, **kwargs):
        super().__init__(**kwargs)
        self._nuisance = nuisance

    @property
    def nuisance(self):
        return self._nuisance

    @nuisance.setter
    def nuisance(self, nuisance):
        self._nuisance = nuisance

    def stat_sum(self):
        wstat = super().stat_sum()
        liketotal = wstat
        if self.nuisance:
            liketotal += self.nuisance_f()
        return liketotal

    def stat_sum_no_prior(self):
        return super().stat_sum()

    def nuisance_f(self):
        for model in self.models:
            if 'PD' in model.name:
                int_PD = model.spectral_model.integral(self.nuisance['e_int_min'], self.nuisance['e_int_max']).to_value('cm-2 s-1')
            elif 'IC' in model.name:
                int_IC = model.spectral_model.integral(self.nuisance['e_int_min'], self.nuisance['e_int_max']).to_value('cm-2 s-1')
        f = int_PD / (int_PD + int_IC)
        f_target = self.nuisance['f']
        f_err = self.nuisance['f_err']
        scale = self.nuisance['scale']
        power = self.nuisance['power']
        return scale * (f-f_target)**power / f_err**power
