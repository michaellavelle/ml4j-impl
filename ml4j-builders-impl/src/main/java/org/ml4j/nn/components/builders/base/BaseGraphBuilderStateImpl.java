/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.components.builders.base;

import org.ml4j.Matrix;
import org.ml4j.nn.components.builders.BaseGraphBuilderState;
import org.ml4j.nn.components.builders.axons.UncompletedFullyConnectedAxonsBuilder;
import org.ml4j.nn.components.builders.componentsgraph.ComponentsGraphNeurons;
import org.ml4j.nn.components.builders.synapses.SynapsesAxonsGraphBuilder;
import org.ml4j.nn.neurons.Neurons;

public class BaseGraphBuilderStateImpl implements BaseGraphBuilderState {
	
	protected ComponentsGraphNeurons<Neurons> componentsGraphNeurons;
	protected Matrix connectionWeights;
	protected Matrix biases;
	protected UncompletedFullyConnectedAxonsBuilder<?> fullyConnectedAxonsBuilder;
	protected SynapsesAxonsGraphBuilder<?, ?> synapsesBuilder;
	
	public BaseGraphBuilderStateImpl(Neurons initialNeurons) {
		this.componentsGraphNeurons = new ComponentsGraphNeuronsImpl<>(initialNeurons);
	}
	
	public BaseGraphBuilderStateImpl() {
	}
	
	@Override
	public ComponentsGraphNeurons<Neurons> getComponentsGraphNeurons() {
		return componentsGraphNeurons;
	}
	
	public void setComponentsGraphNeurons(ComponentsGraphNeurons<Neurons> componentsGraphNeurons) {
		this.componentsGraphNeurons = componentsGraphNeurons;
	}
	
	@Override
	public Matrix getConnectionWeights() {
		return connectionWeights;
	}
	
	@Override
	public void setConnectionWeights(Matrix connectionWeights) {
		this.connectionWeights = connectionWeights;
	}
	
	@Override
	public Matrix getBiases() {
		return biases;
	}

	@Override
	public void setBiases(Matrix biases) {
		this.biases = biases;
	}

	@Override
	public UncompletedFullyConnectedAxonsBuilder<?> getFullyConnectedAxonsBuilder() {
		return fullyConnectedAxonsBuilder;
	}
	
	@Override
	public void setFullyConnectedAxonsBuilder(UncompletedFullyConnectedAxonsBuilder<?> fullyConnectedAxonsBuilder) {
		this.fullyConnectedAxonsBuilder = fullyConnectedAxonsBuilder;
	}
	
	@Override
	public SynapsesAxonsGraphBuilder<?, ?> getSynapsesBuilder() {
		return synapsesBuilder;
	}
	
	@Override
	public void setSynapsesBuilder(SynapsesAxonsGraphBuilder<?, ?> synapsesBuilder) {
		this.synapsesBuilder = synapsesBuilder;
	}
}
