import cruising
import polardiagram
import processing
from .wind import true_wind_to_apparent, apparent_wind_to_true

__pdoc__ = {
    "hrosailing.wind.set_resolution": False,
    "hrosailing.wind.convert_wind": False,
    "hrosailing.polardiagram.PolarDiagramTable.__getitem__": True,
    "hrosailing.polardiagram.PolarDiagramTable.__str__": True,
    "hrosailing.polardiagram.PolarDiagramTable.__repr__": True,
    "hrosailing.polardiagram.PolarDiagramMultiSails.__getitem__": True,
    "hrosailing.polardiagram.PolarDiagramMultiSails.__str__": True,
    "hrosailing.polardiagram.PolarDiagramMultiSails.__repr__": True,
    "hrosailing.polardiagram.PolarDiagramCurve.__repr__": True,
    "hrosailing.polardiagram.PolarDiagramCurve.__call__": True,
    "hrosailing.polardiagram.PolarDiagramPointcloud.__str__": True,
    "hrosailing.polardiagram.PolarDiagramPointcloud.__repr__": True,
    "hrosailing.processing.pipeline.PolarPipeline.__call__": True,
    "hrosailing.processing.pipelinecomponents.interpolator.gauss_potential": False,
    "hrosailing.processing.pipelinecomponents.interpolator.scaled": False,
    "hrosailing.processing.pipelinecomponents.interpolator.euclidean_norm": False,
    "hrosailing.processing.pipelinecomponents.neighbourhood.euclidean_norm": False,
    "hrosailing.processing.pipelinecomponents.neighbourhood.scaled": False,
    "hrosailing.processing.pipelinecomponents.sampler.make_circle": False,
    "hrosailing.processing.pipelinecomponents.weigher.euclidean_norm": False,
    "hrosailing.processing.pipelinecomponents.weigher.scaled": False,
    "hrosailing.processing.pipelinecomponents.weigher.WeightedPoints.__getitem__": True,
    "hrosailing.processing.pipelinecomponents.weigher.CylindricMeanWeigher.__repr__": True,
    "hrosailing.processing.pipelinecomponents.weigher.CylindricMemberWeigher.__repr__": True,
}
