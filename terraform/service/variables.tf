variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-3"
}

variable "service_name" {
  description = "App Runner service name"
  type        = string
  default     = "seattle-energy-api"
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

variable "environment" {
  description = "Deployment environment (e.g. dev, prod)"
  type        = string
  default     = "prod"
}
