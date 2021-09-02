from scipy import stats


class CpModel:
    def __init__(self, intensity):
        self.intensity = intensity
        self.cp_pmf = self.create_distribution()

    def create_distribution(self):
        return stats.geom(self.intensity)
    
    def pmf(self, k):
        return self.cp_pmf.pmf(k)
    
    def pmf_0(self, k):
        if k == 0:
            return 1
        else:
            return 0

    def hazard(self, k):
        return 1.0/self.intensity
        
    def hazard_vector(self, k, m):
        return 1.0/self.intensity
    















