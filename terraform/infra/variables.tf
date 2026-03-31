variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-3"
}

variable "repo_name" {
  description = "ECR repository name"
  type        = string
  default     = "seattle-energy-predictor"
}

variable "environment" {
  description = "Deployment environment (e.g. dev, prod)"
  type        = string
  default     = "prod"
}
