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
package org.ml4j.nn.components.onetoone;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.ml4j.nn.components.NeuralComponentType;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultChainableDirectedComponentAdapter<A extends DefaultChainableDirectedComponentActivation, C> implements DefaultChainableDirectedComponent<A, C> {

	/**
	 * Default serialization.
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultChainableDirectedComponentAdapter.class);
	
	protected DefaultChainableDirectedComponent<A, C> delegated;
	
	private static Map<String, AtomicLong> timesByClassName;
	private static Map<String, AtomicInteger> countsByClassName;
	protected String name;
	
	static {
		timesByClassName = new HashMap<>();
		countsByClassName = new HashMap<>();
	}
	
	
	public DefaultChainableDirectedComponentAdapter(DefaultChainableDirectedComponent<A, C> delegated, String name) {
		this.delegated = delegated;
		this.name = name;
	}

	@Override
	public C getContext(DirectedComponentsContext directedComponentsContext, int componentIndex) {
		return delegated.getContext(directedComponentsContext, componentIndex);
	}
	
	public static void printTimes() {
		
		System.out.println("\n");
		System.out.println("Totals Times by Component:\n");
		for (Entry<String, AtomicLong> entry : timesByClassName.entrySet()) {
			System.out.println(entry.getKey() + ":" + entry.getValue().get());
		}
		System.out.println("\nAverage Times By Component:\n");
		for (Entry<String, AtomicLong> entry : timesByClassName.entrySet()) {
			int count = countsByClassName.get(entry.getKey()).get();
			System.out.println(entry.getKey() + ":" + entry.getValue().get() / count);
		}
		
	}
	
	
	public static void addTime(long timeTaken, String name) {
		AtomicLong existingTime = timesByClassName.get(name);
		if (existingTime == null) {
			existingTime = new AtomicLong(0);
			timesByClassName.put(name, existingTime);
		} 
		AtomicInteger existingCount = countsByClassName.get(name);
		if (existingCount == null) {
			existingCount = new AtomicInteger(0);
			countsByClassName.put(name, existingCount);

		} 
		existingTime.addAndGet(timeTaken);
		existingCount.addAndGet(1);
	}

	@Override
	public A forwardPropagate(NeuronsActivation input, C context) {
		LOGGER.info(getComponentType().toString());
		long startTime = new Date().getTime();
		A activation =delegated.forwardPropagate(input, context);
		long endTime = new Date().getTime();
		long timeTaken = endTime - startTime;
		addTime(timeTaken, "fp:" + name);
		return activation;
	}

	@Override
	public NeuralComponentType<? extends DefaultChainableDirectedComponent<?, ?>> getComponentType() {
		return delegated.getComponentType();
	}

	@Override
	public DefaultChainableDirectedComponent<A, C> dup() {
		return new DefaultChainableDirectedComponentAdapter<>(delegated.dup(), name);
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return delegated.decompose();
	}

	@Override
	public Neurons getInputNeurons() {
		return delegated.getInputNeurons();
	}

	@Override
	public Neurons getOutputNeurons() {
		return delegated.getOutputNeurons();
	}

}
