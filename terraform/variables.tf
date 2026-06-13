variable "do_token" {
  description = "Token DO"
  type        = string
  sensitive   = true        # ne pas afficher dans les logs
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "fra1"          # valeur par défaut, si non fournie
}

variable "ssh_key_name" {
  description = "Name of the SSH key"
  type        = string
  default     = "my-ssh-key"
}