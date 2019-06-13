/*
 * Copyright 2017 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ml4j.nn.axons;

import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapses.
 * 
 * @author Michael Lavelle
 */
public class DirectedAxonsComponentImpl<L extends Neurons, R extends Neurons> 
    implements DirectedAxonsComponent<L, R> {
  
  /**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

    private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedAxonsComponentImpl.class);
  
  private Axons<? extends L, ? extends R, ?> axons;
  
  /**
   * Create a new implementation of DirectedSynapses.
   * 
   * @param axons The Axons within these synapses
   * @param activationFunction The activation function within these synapses
   */
  public DirectedAxonsComponentImpl(Axons<? extends L, ? extends R, ?> axons) {
    super();
    this.axons = axons;
  }

  @Override
  public Axons<? extends L, ? extends R, ?> getAxons() {
    return axons;
  }

  @Override
  public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation inputNeuronsActivation,
      AxonsContext axonsContext) {
      
    LOGGER.debug("Forward propagating through DirectedAxonsComponent");
    AxonsActivation axonsActivation = 
        axons.pushLeftToRight(inputNeuronsActivation, null, 
            axonsContext);   
    
    return new DirectedAxonsComponentActivationImpl(axonsActivation, axonsContext);
  
  }

@Override
public AxonsContext getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
	return directedComponentsContext.getContext(this, () -> new AxonsContextImpl(directedComponentsContext.getMatrixFactory(), false));
}

@Override
public List<ChainableDirectedComponent<NeuronsActivation, ? extends ChainableDirectedComponentActivation<NeuronsActivation>, ?>> decompose() {
	return Arrays.asList(this);
}

  //@Override
  //public DirectedAxonsComponent<L, R> dup() {
//	return new DirectedAxonsComponentImpl<>(axons.dup());
  //}
}
