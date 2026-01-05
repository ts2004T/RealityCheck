import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Get the root directory (parent of server/)
const rootDir = path.join(__dirname, '..');

console.log("Starting RealityCheck (Flask App)...");

// Spawn the Python process from the root directory
const python = spawn('python', ['app.py'], { 
  stdio: 'inherit',
  cwd: rootDir 
});

python.on('error', (err) => {
  console.error('Failed to start Python process:', err);
});

python.on('close', (code) => {
  console.log(`Python process exited with code ${code}`);
  process.exit(code || 0);
});

// Handle termination signals
process.on('SIGTERM', () => {
  python.kill('SIGTERM');
});

process.on('SIGINT', () => {
  python.kill('SIGINT');
});
