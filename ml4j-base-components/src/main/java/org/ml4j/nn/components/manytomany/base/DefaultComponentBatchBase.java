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
package org.ml4j.nn.components.manytomany.base;

import java.util.List;

import org.ml4j.nn.components.NeuralComponentBaseType;
import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentBatch;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;

/**
 * Default base class for a batch of DefaultDirectedComponentChain instances that can be activated in parallel.
 * 
 * @author Michael Lavelle
 */
public abstract class DefaultComponentBatchBase implements DefaultDirectedComponentBatch {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	protected List<DefaultChainableDirectedComponent<?, ?>> parallelComponents;

	public DefaultComponentBatchBase(List<DefaultChainableDirectedComponent<?, ?>> parallelComponents) {
		this.parallelComponents = parallelComponents;
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> getComponents() {
		return parallelComponents;
	}

	@Override
	public NeuralComponentType<?> getComponentType() {
		return NeuralComponentType.getBaseType(NeuralComponentBaseType.COMPONENT_BATCH);
	}

}
