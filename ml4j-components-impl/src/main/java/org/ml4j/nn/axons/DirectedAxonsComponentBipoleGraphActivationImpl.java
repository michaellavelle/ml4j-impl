package org.ml4j.nn.axons;

import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentChainActivation;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphActivationImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedAxonsComponentBipoleGraphActivationImpl extends DirectedComponentsBipoleGraphActivationImpl<NeuronsActivation, DirectedComponentChainActivation<NeuronsActivation, DirectedAxonsComponentActivation>>
	implements DirectedAxonsComponentBipoleGraphActivation {

	public DirectedAxonsComponentBipoleGraphActivationImpl(
			OneToManyDirectedComponentActivation<NeuronsActivation> inputLinkActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DirectedComponentChainActivation<NeuronsActivation, DirectedAxonsComponentActivation>> edgesActivation,
			ManyToOneDirectedComponentActivation<NeuronsActivation> outputLinkActivation) {
		super(inputLinkActivation, edgesActivation, outputLinkActivation);
	}

	  public double getAverageRegularistationCost(DirectedComponentsContext context) {
	    return getTotalRegularisationCost(context)
	        / outputLinkActivation.getOutput().getActivations().getRows();
	  }

	  public double getTotalRegularisationCost(DirectedComponentsContext context) {
	    double totalRegularisationCost = 0d;
	    for (DirectedAxonsComponentActivation activation : edgesActivation.getActivations().stream().flatMap(a -> a.getActivations().stream()).collect(Collectors.toList())) {
	      AxonsContext axonsContext = context.getContext(activation.getAxonsComponent(), () -> new AxonsContextImpl(context.getMatrixFactory(), false));
	      totalRegularisationCost =
	          totalRegularisationCost + activation.getTotalRegularisationCost(axonsContext);
	    }
	    return totalRegularisationCost;
	  
	  }
	

}
