package org.ml4j.nn.components;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.ml4j.nn.axons.AxonsGradient;

public class DirectedComponentChainActivationImpl<I, A extends ChainableDirectedComponentActivation<I>> implements DirectedComponentChainActivation<I, A> {

	private I output;
	private List<A> activations;
	
	public DirectedComponentChainActivationImpl(List<A> activations, I output) {
		this.activations = activations;
		this.output = output;
	}
	
	@Override
	public DirectedComponentGradient<I> backPropagate(DirectedComponentGradient<I> outerGradient) {
		List<ChainableDirectedComponentActivation<I>> reversedSynapseActivations =
			        new ArrayList<>();
			    reversedSynapseActivations.addAll(activations);
			    Collections.reverse(reversedSynapseActivations);
			    return backPropagateAndAddToSynapseGradientList(outerGradient,
			        reversedSynapseActivations);
	}
	
	private DirectedComponentGradient<I> backPropagateAndAddToSynapseGradientList(
		      DirectedComponentGradient<I> outerSynapsesGradient,
		      List<ChainableDirectedComponentActivation<I>> activationsToBackPropagateThrough) {

			List<AxonsGradient> totalTrainableAxonsGradients = new ArrayList<>();
			totalTrainableAxonsGradients.addAll(outerSynapsesGradient.getTotalTrainableAxonsGradients());
			
		    DirectedComponentGradient<I> finalGrad = outerSynapsesGradient;
		    DirectedComponentGradient<I> synapsesGradient = outerSynapsesGradient;
		    for (ChainableDirectedComponentActivation<I> synapsesActivation : activationsToBackPropagateThrough) {
		     
		      synapsesGradient =
		          synapsesActivation.backPropagate(synapsesGradient);
		      
		      totalTrainableAxonsGradients.addAll(synapsesGradient.getTotalTrainableAxonsGradients());
		      finalGrad = synapsesGradient;

		    }

		    return new DirectedComponentGradientImpl<>(totalTrainableAxonsGradients, finalGrad.getOutput());
		  }


	@Override
	public List<A> getActivations() {
		return activations;
	}

	@Override
	public I getOutput() {
		return output;
	}

	
}
