# Homebrew formula skeleton for the Autonomous Financial Agent War Room.
#
# Usage (once published to PyPI):
#   1. brew tap metehan05-eng/warroom https://github.com/metehan05-eng/homebrew-warroom
#   2. brew install financial-agent-war-room
#
# Before submitting to a tap, fill in `url`/`sha256` from the PyPI release and
# run `brew update-python-resources financial-agent-war-room` to generate the
# `resource` blocks for the Python dependencies.
class FinancialAgentWarRoom < Formula
  include Language::Python::Virtualenv

  desc "Multi-agent market debate in your terminal, decided by a Chief Risk Officer"
  homepage "https://github.com/metehan05-eng/Autonomous-Financial-Agent-War-Room"
  # TODO: point at the PyPI sdist and paste its sha256:
  url "https://files.pythonhosted.org/packages/source/f/financial-agent-war-room/financial_agent_war_room-0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  head "https://github.com/metehan05-eng/Autonomous-Financial-Agent-War-Room.git", branch: "main"

  depends_on "python@3.12"

  # `brew update-python-resources financial-agent-war-room` inserts the
  # vendored dependency `resource` blocks here.

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "multi-agent", shell_output("#{bin}/financial-warroom --help")
  end
end
