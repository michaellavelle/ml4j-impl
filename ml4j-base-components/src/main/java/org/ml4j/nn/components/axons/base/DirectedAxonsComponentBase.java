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
package org.ml4j.nn.components.axons.base;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.axons.Axons;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsContextImpl;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.axons.DirectedAxonsComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.neurons.Neurons;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default base class for a DirectedAxonsComponent - a DefaultChainableDirectedComponent adapter for an Axons instance.
 * 
 * 
 * @author Michael Lavelle
 *
 * @param <L> The type of Neurons on the LHS of this DirectedAxonsComponent.
 * @param <R> The type of Neurons on the RHS of this DirectedAxonsComponent.
 * @param <A> The specific type of Axons wrapped by this DirectedAxonsComponent.
 */
public abstract class DirectedAxonsComponentBase<L extends Neurons, R extends Neurons, A extends Axons<? extends L, ? extends R,  ?>> implements DirectedAxonsComponent<L, R, A> {

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(DirectedAxonsComponentBase.class);

	private static final String DIRECTED_AXONS_SUBTYPE_NAME = "DIRECTED";	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	protected A axons;
	
	/**
	 * @param axons The axons instance wrapped by this DirectedAxonsComponent.
	 */
	public DirectedAxonsComponentBase(A axons) {
		this.axons = axons;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public AxonsContext getContext(DirectedComponentsContext context, int componentIndex) {
		return context.getContext(this, () -> new AxonsContextImpl(context.getMatrixFactory(), context.isTrainingContext(), false));
	}

	@Override
	public A getAxons() {
		return axons;
	}

	@Override
	public NeuralComponentType<DirectedAxonsComponent<L, R, A>> getComponentType() {
		return NeuralComponentType.createSubType(NeuralComponentType.createSubType(NeuralComponentType.getBaseType(NeuralComponentBaseType.AXONS), 
				DIRECTED_AXONS_SUBTYPE_NAME), axons.getClass().getSimpleName());
	}

	@Override
	public Neurons getInputNeurons() {
		return axons.getLeftNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return axons.getRightNeurons();
	}

}
