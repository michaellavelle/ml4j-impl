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

package org.ml4j.nn.synapses;

import org.ml4j.Matrix;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsContextImpl;
import org.ml4j.nn.axons.DirectedAxonsComponent;
import org.ml4j.nn.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.graph.DirectedDipoleGraph;
import org.ml4j.nn.graph.DirectedPath;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DirectedSynapsesActivation.
 * 
 * @author Michael Lavelle
 */
public abstract class DirectedSynapsesActivationBase implements DirectedSynapsesActivation {

 private static final Logger LOGGER = 
      LoggerFactory.getLogger(DirectedSynapsesActivationBase.class);
  
  protected NeuronsActivation inputActivation;
  protected DirectedDipoleGraph<DirectedAxonsComponentActivation> axonsActivationGraph;
  protected DifferentiableActivationFunctionActivation activationFunctionActivation;
  protected NeuronsActivation outputActivation;
  protected DirectedSynapses<?, ?> synapses;
  protected DirectedComponentsContext synapsesContext;
  
  /**
   * Construct a new default DirectedSynapsesActivation
   * 
   * @param synapses The DirectedSynapses
   * @param inputActivation The input NeuronsActivation of the DirectedSynapses
   *        following a forward propagation
   * @param axonsActivationGraph The axons NeuronsActivation graph of the DirectedSynapses
   *        following a forward propagation    
   * @param outputActivation The output NeuronsActivation of the DirectedSynapses
   *        following a forward propagation.
   */
  public DirectedSynapsesActivationBase(DirectedSynapses<?, ?> synapses, 
      NeuronsActivation inputActivation, 
      DirectedDipoleGraph<DirectedAxonsComponentActivation> axonsActivationGraph, 
      DifferentiableActivationFunctionActivation activationFunctionActivation,
      NeuronsActivation outputActivation, DirectedComponentsContext synapsesContext) {
    this.inputActivation = inputActivation;
    this.outputActivation = outputActivation;
    this.synapses = synapses;
    this.axonsActivationGraph = axonsActivationGraph;
    this.activationFunctionActivation = activationFunctionActivation;
    this.synapsesContext = synapsesContext;
  }
  
  @Override
  public NeuronsActivation getOutput() {
    return outputActivation;
  }

  @Override
  public DirectedSynapses<?, ?> getSynapses() {
    return synapses;
  }
  /*
  @Override
  public double getAverageRegularisationCost(DirectedSynapsesContext synapsesContext) {
    LOGGER.debug("Calculating average regularisation cost");
    return getTotalRegularisationCost(synapsesContext) 
        / outputActivation.getActivations().getRows();
  }
  */

  @Override
  public double getTotalRegularisationCost(DirectedComponentsContext synapsesContext) {
  
    double totalRegularisationCost = 0d;
    int pathIndex = 0;
    for (DirectedPath<DirectedAxonsComponent<?, ?>> parallelAxonsPath  : 
          synapses.getAxonsGraph().getParallelPaths()) {
        
      int axonsIndex = 0;
      for (DirectedAxonsComponent<?, ?> axonsComponent : parallelAxonsPath.getEdges()) {
        AxonsContext axonsContext = synapsesContext.getContext(axonsComponent, () -> new AxonsContextImpl(synapsesContext.getMatrixFactory(), false));

        if (axonsContext.getRegularisationLambda() != 0) {

          LOGGER.debug("Calculating total regularisation cost");
          
          Matrix weightsWithBiases = axonsComponent.getAxons().getDetachedConnectionWeights();

          int[] rows = new int[weightsWithBiases.getRows()
              - (this.getSynapses().getLeftNeurons().hasBiasUnit() ? 1 : 0)];
          int[] cols = new int[weightsWithBiases.getColumns()
              - (this.getSynapses().getRightNeurons().hasBiasUnit() ? 1 : 0)];
          for (int j = 0; j < weightsWithBiases.getColumns(); j++) {
            cols[j - (this.getSynapses().getRightNeurons().hasBiasUnit() ? 1 : 0)] = j;
          }
          for (int j = 1; j < weightsWithBiases.getRows(); j++) {
            rows[j - (this.getSynapses().getLeftNeurons().hasBiasUnit() ? 1 : 0)] = j;
          }

          Matrix weightsWithoutBiases = weightsWithBiases.get(rows, cols);

          double regularisationMatrix = weightsWithoutBiases.mul(weightsWithoutBiases).sum();
          totalRegularisationCost = 
              totalRegularisationCost 
              + ((axonsContext.getRegularisationLambda()) * regularisationMatrix) / 2;
        }
        
        
        axonsIndex++;
      }
      pathIndex++;
      
    }
    return totalRegularisationCost;
  }
  

  /*
  @Override
  public DirectedDipoleGraph<AxonsActivation> getAxonsActivationGraph() {
    return axonsActivationGraph;
  }

  @Override
  public NeuronsActivation getInput() {
    return inputActivation;
  }
  
  @Override
  public DifferentiableActivationFunctionActivation getActivationFunctionActivation() {
    return activationFunctionActivation;
  }
  
  
  @Override
	public DirectedSynapsesContext getContext() {
		return synapsesContext;
	}
	
	  */
  
}
