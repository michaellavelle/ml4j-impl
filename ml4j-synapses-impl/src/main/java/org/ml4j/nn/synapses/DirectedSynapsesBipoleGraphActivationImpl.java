package org.ml4j.nn.synapses;

import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentsBipoleGraphActivationImpl;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.OneToManyDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DirectedSynapsesBipoleGraphActivationImpl extends DirectedComponentsBipoleGraphActivationImpl<NeuronsActivation, DirectedSynapsesChainActivation>
	implements DirectedSynapsesBipoleGraphActivation {

	public DirectedSynapsesBipoleGraphActivationImpl(
			OneToManyDirectedComponentActivation<NeuronsActivation> inputLinkActivation,
			DirectedComponentBatchActivation<NeuronsActivation, DirectedSynapsesChainActivation> edgesActivation,
			ManyToOneDirectedComponentActivation<NeuronsActivation> outputLinkActivation) {
		super(inputLinkActivation, edgesActivation, outputLinkActivation);
	}

	  public double getAverageRegularistationCost(DirectedComponentsContext context) {
	    return getTotalRegularisationCost(context)
	        / outputLinkActivation.getOutput().getActivations().getRows();
	  }

	  public double getTotalRegularisationCost(DirectedComponentsContext context) {
	    double totalRegularisationCost = 0d;
	    for (DirectedSynapsesActivation activation : edgesActivation.getActivations().stream().flatMap(a -> a.getActivations().stream()).collect(Collectors.toList())) {
	      //DirectedSynapsesContext synapsesContext = context.getContext(activation.getSynapses(), () -> new DirectedSynapsesContextImpl(context.getMatrixFactory(), false));
	      totalRegularisationCost =
	          totalRegularisationCost + activation.getTotalRegularisationCost(context);
	    }
	    return totalRegularisationCost;
	  
	  }
	

}
