function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%======================INIT====================================================
%Codifies the y labels into to a matrix where each row is vector of 0s and 1,
%and the position of the 1 is indicated by the y value of each training example
I = eye(num_labels);
Y = I(y,:);
OnesV = ones(m,1);
%=======================PART1==================================================

%layer1
a1 = X; %Without bias Unit
A1 = [OnesV a1]; % Add the bias unit
%layer2
a2 = sigmoid(A1*Theta1'); %Without bias Unit
A2 = [OnesV a2]; % Add the bias unit
%layer3
a3 = sigmoid(A2*Theta2'); % No need to add bias unit as it is the last layer
h = a3; %note this is a matrix mxnumblayers


% Cost without regularization:
J = sum(sum((-Y.* log(h) - (1-Y).*log(1-h))))/m;
% Cost with regularization:
L=(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))*(lambda/(2*m));
J=J+L;

%=======================PART2==================================================
D1 = zeros(size(Theta1)); % 25 x 401
D2 = zeros(size(Theta2)); % 10 x 26

for t=1:m,
 %1. Foward propagation for example t
  a1 = X(t,:); % 1x400
  A1 = [1 a1]; % 1x401
  
  z2 = A1*Theta1';   % (1x401) * (25x401)' = (1x25)
  a2 = sigmoid(z2);  % 1x25
  A2 = [1 a2];       % 1x26
  
  z3 = A2*Theta2';     % (1x26) * (10*26)' = (1x10)
  a3 = sigmoid(z3);   % (1x10)
  h=a3;               % (1x10)
  
  %2.Error calculations
  d3 = h - Y(t,:);    % (1x10) - (1x10) = (1x10)
  d2 = d3 * Theta2(:,2:end).* sigmoidGradient(z2); % (1x10) * (10x25) .* (1x25) = (1x25) 
  
  %3.Accumulate
  D1 = D1 + d2' * A1; %(25x1) * (1x401) = (25x401)
  D2 = D2 + d3' * A2; %(10x1) * (1x26) = (10x26)
  
end;

L1 = (lambda/m) * [zeros(size(Theta1)(1),1) , Theta1(:,2:end)];
L2 = (lambda/m) * [zeros(size(Theta2)(1),1) , Theta2(:,2:end)];

Theta1_grad = D1/m + L1; 
Theta2_grad = D2/m + L2;

%grad = ((h-y)'*X)/m + (lambda/m)*([[0];theta(2:n,1)])';


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
