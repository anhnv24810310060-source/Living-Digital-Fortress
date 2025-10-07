/**
 * ShieldX Camouflage Edge Worker
 * Production-ready Cloudflare Worker for adaptive camouflage
 */

const ORCHESTRATOR_ENDPOINT = 'https://orchestrator.shieldx.internal';
const CAMOUFLAGE_API_KEY = 'CAMOUFLAGE_API_KEY'; // Environment variable
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const MAX_REQUESTS_PER_WINDOW = 100;

// Reconnaissance detection patterns
const RECON_PATTERNS = {
  nmap: [
    /nmap/i,
    /masscan/i,
    /zmap/i
  ],
  nikto: [
    /nikto/i,
    /whisker/i
  ],
  dirb: [
    /dirb/i,
    /dirbuster/i,
    /gobuster/i,
    /ffuf/i
  ],
  sqlmap: [
    /sqlmap/i,
    /havij/i
  ],
  burp: [
    /burp/i,
    /professional/i
  ],
  generic: [
    /scanner/i,
    /crawler/i,
    /spider/i,
    /bot/i,
    /automated/i
  ]
};

// Suspicious path patterns
const SUSPICIOUS_PATHS = [
  /\/admin/i,
  /\/wp-admin/i,
  /\/phpmyadmin/i,
  /\/\.git/i,
  /\/\.env/i,
  /\/config/i,
  /\/backup/i,
  /\/test/i,
  /\/debug/i,
  /\/trace\.axd/i,
  /\/server-status/i,
  /\/nginx_status/i,
  /\.\./,
  /%2e%2e/i,
  /\/etc\/passwd/i,
  /\/proc\/version/i
];

class CamouflageWorker {
  constructor() {
    this.rateLimitMap = new Map();
  }

  async handleRequest(request) {
    try {
      const startTime = Date.now();
      const clientIP = request.headers.get('CF-Connecting-IP') || 'unknown';
      const userAgent = request.headers.get('User-Agent') || '';
      const url = new URL(request.url);

      // Rate limiting
      if (!this.checkRateLimit(clientIP)) {
        return new Response('Rate limit exceeded', { 
          status: 429,
          headers: { 'Retry-After': '60' }
        });
      }

      // Detect reconnaissance attempt
      const reconType = this.detectReconnaissance(request, userAgent, url.pathname);
      
      if (reconType) {
        // Log reconnaissance attempt
        await this.logReconAttempt(clientIP, userAgent, url.pathname, reconType);
        
        // Get appropriate camouflage template
        const template = await this.getCamouflageTemplate(reconType, clientIP, userAgent);
        
        if (template) {
          // Apply camouflage and return deceptive response
          return this.applyCamouflage(request, template, reconType);
        }
      }

      // Normal request - pass through or apply default template
      return this.handleNormalRequest(request);

    } catch (error) {
      console.error('Camouflage Worker error:', error);
      return new Response('Internal Server Error', { status: 500 });
    }
  }

  detectReconnaissance(request, userAgent, pathname) {
    // Check User-Agent patterns
    for (const [type, patterns] of Object.entries(RECON_PATTERNS)) {
      for (const pattern of patterns) {
        if (pattern.test(userAgent)) {
          return type;
        }
      }
    }

    // Check suspicious paths
    for (const pattern of SUSPICIOUS_PATHS) {
      if (pattern.test(pathname)) {
        return 'path_probe';
      }
    }

    // Check request headers for scanning tools
    const headers = request.headers;
    if (headers.get('X-Scanner') || 
        headers.get('X-Forwarded-For')?.includes('scanner') ||
        headers.get('Accept')?.includes('application/json') && pathname.includes('api')) {
      return 'api_probe';
    }

    // Check for rapid sequential requests (potential scan)
    const clientIP = request.headers.get('CF-Connecting-IP');
    if (this.isRapidScanning(clientIP)) {
      return 'rapid_scan';
    }

    return null;
  }

  async getCamouflageTemplate(reconType, clientIP, userAgent) {
    try {
      // Determine appropriate template based on reconnaissance type
      let templateName = this.selectTemplate(reconType, userAgent);
      
      const response = await fetch(`${ORCHESTRATOR_ENDPOINT}/v1/camouflage/template/${templateName}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${CAMOUFLAGE_API_KEY}`,
          'Content-Type': 'application/json',
          'X-Client-IP': clientIP,
          'X-Recon-Type': reconType
        }
      });

      if (response.ok) {
        return await response.json();
      } else {
        console.error('Failed to get template:', response.status);
        return this.getFallbackTemplate(templateName);
      }
    } catch (error) {
      console.error('Template fetch error:', error);
      return this.getFallbackTemplate('apache');
    }
  }

  selectTemplate(reconType, userAgent) {
    // Smart template selection based on reconnaissance type and user agent
    if (userAgent.includes('nmap') || reconType === 'nmap') {
      return 'apache'; // Apache is commonly targeted
    }
    
    if (userAgent.includes('nikto') || reconType === 'nikto') {
      return 'nginx'; // Nginx for web vulnerability scanners
    }
    
    if (reconType === 'api_probe') {
      return 'nginx'; // Modern API servers often use Nginx
    }
    
    if (reconType === 'path_probe') {
      return 'iis'; // IIS for Windows-specific probes
    }
    
    // Default to Apache (most common)
    return 'apache';
  }

  applyCamouflage(request, template, reconType) {
    const url = new URL(request.url);
    const pathname = url.pathname;

    // Apply template headers
    const headers = new Headers();
    for (const [key, value] of Object.entries(template.headers || {})) {
      headers.set(key, this.interpolateVariables(value, request));
    }

    // Determine response based on path and template
    let statusCode = 200;
    let body = '';
    let contentType = 'text/html';

    // Handle error pages
    if (this.isSuspiciousPath(pathname)) {
      if (template.error_pages && template.error_pages['403']) {
        statusCode = 403;
        body = template.error_pages['403'].body;
        contentType = template.error_pages['403'].content_type || 'text/html';
      } else {
        statusCode = 403;
        body = this.generateGenericErrorPage(403, template.name);
      }
    } else if (pathname.includes('nonexistent') || pathname.endsWith('.php')) {
      if (template.error_pages && template.error_pages['404']) {
        statusCode = 404;
        body = template.error_pages['404'].body;
        contentType = template.error_pages['404'].content_type || 'text/html';
      } else {
        statusCode = 404;
        body = this.generateGenericErrorPage(404, template.name);
      }
    } else {
      // Generate convincing default page
      body = this.generateDefaultPage(template);
    }

    // Apply behavioral timing
    const delay = this.calculateDelay(template.behavioral_patterns?.response_timing);
    
    // Set additional headers
    headers.set('Content-Type', contentType);
    headers.set('Date', new Date().toUTCString());
    
    // Add template-specific headers
    if (template.name === 'apache') {
      headers.set('Accept-Ranges', 'bytes');
    } else if (template.name === 'nginx') {
      headers.set('Accept-Ranges', 'bytes');
    } else if (template.name === 'iis') {
      headers.set('X-Powered-By', 'ASP.NET');
    }

    // Interpolate variables in body
    body = this.interpolateVariables(body, request);

    // Simulate response delay
    if (delay > 0) {
      return new Promise(resolve => {
        setTimeout(() => {
          resolve(new Response(body, { status: statusCode, headers }));
        }, delay);
      });
    }

    return new Response(body, { status: statusCode, headers });
  }

  generateDefaultPage(template) {
    switch (template.name) {
      case 'apache':
        return `<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head><title>Index of /</title></head>
<body>
<h1>Index of /</h1>
<table>
<tr><th valign="top"><img src="/icons/blank.gif" alt="[ICO]"></th><th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th><th><a href="?C=S;O=A">Size</a></th></tr>
<tr><th colspan="4"><hr></th></tr>
<tr><td valign="top"><img src="/icons/folder.gif" alt="[DIR]"></td><td><a href="cgi-bin/">cgi-bin/</a></td><td align="right">-</td><td>&nbsp;</td></tr>
</table>
<address>Apache/${template.version} Server</address>
</body></html>`;

      case 'nginx':
        return `<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
body { width: 35em; margin: 0 auto; font-family: Tahoma, Verdana, Arial, sans-serif; }
</style>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and working. Further configuration is required.</p>
<p>For online documentation and support please refer to <a href="http://nginx.org/">nginx.org</a>.<br/>
Commercial support is available at <a href="http://nginx.com/">nginx.com</a>.</p>
<p><em>Thank you for using nginx.</em></p>
</body>
</html>`;

      case 'iis':
        return `<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1" />
<title>IIS Windows Server</title>
<style type="text/css">
body{margin:0;font-size:.7em;font-family:Verdana, Arial, Helvetica, sans-serif;background:#EEEEEE;}
</style>
</head>
<body>
<div id="header"><h1>Internet Information Services</h1></div>
<div id="content">
<div class="content-container">
<h2>Welcome</h2>
<p>This is the default web site for this server. The web site content is located in the default directory.</p>
</div>
</div>
</body>
</html>`;

      default:
        return '<html><body><h1>Welcome</h1><p>Server is running.</p></body></html>';
    }
  }

  generateGenericErrorPage(statusCode, serverType) {
    const serverName = serverType === 'apache' ? 'Apache/2.4.54' : 
                      serverType === 'nginx' ? 'nginx/1.22.1' : 
                      'Microsoft-IIS/10.0';

    switch (statusCode) {
      case 403:
        return `<html><head><title>403 Forbidden</title></head><body><h1>Forbidden</h1><p>You don't have permission to access this resource.</p><hr><address>${serverName}</address></body></html>`;
      case 404:
        return `<html><head><title>404 Not Found</title></head><body><h1>Not Found</h1><p>The requested URL was not found on this server.</p><hr><address>${serverName}</address></body></html>`;
      default:
        return `<html><head><title>${statusCode} Error</title></head><body><h1>Error ${statusCode}</h1><hr><address>${serverName}</address></body></html>`;
    }
  }

  interpolateVariables(text, request) {
    const url = new URL(request.url);
    const now = new Date();
    
    const replacements = {
      '{{host}}': url.hostname,
      '{{path}}': url.pathname,
      '{{port}}': url.port || (url.protocol === 'https:' ? '443' : '80'),
      '{{client_ip}}': request.headers.get('CF-Connecting-IP') || 'unknown',
      '{{timestamp}}': now.toISOString(),
      '{{date}}': now.toDateString(),
      '{{time}}': now.toTimeString()
    };

    let result = text;
    for (const [placeholder, value] of Object.entries(replacements)) {
      result = result.replace(new RegExp(placeholder.replace(/[{}]/g, '\\$&'), 'g'), value);
    }

    return result;
  }

  calculateDelay(timing) {
    if (!timing) return 0;
    
    const min = timing.min_ms || 50;
    const max = timing.max_ms || 200;
    const jitter = timing.jitter_factor || 0.1;
    
    let delay = min + Math.random() * (max - min);
    delay *= (1 + (Math.random() - 0.5) * jitter);
    
    return Math.max(0, Math.min(delay, max));
  }

  isSuspiciousPath(pathname) {
    return SUSPICIOUS_PATHS.some(pattern => pattern.test(pathname));
  }

  checkRateLimit(clientIP) {
    const now = Date.now();
    const windowStart = now - RATE_LIMIT_WINDOW;
    
    if (!this.rateLimitMap.has(clientIP)) {
      this.rateLimitMap.set(clientIP, []);
    }
    
    const requests = this.rateLimitMap.get(clientIP);
    
    // Remove old requests outside the window
    while (requests.length > 0 && requests[0] < windowStart) {
      requests.shift();
    }
    
    // Check if limit exceeded
    if (requests.length >= MAX_REQUESTS_PER_WINDOW) {
      return false;
    }
    
    // Add current request
    requests.push(now);
    return true;
  }

  isRapidScanning(clientIP) {
    if (!this.rateLimitMap.has(clientIP)) {
      return false;
    }
    
    const requests = this.rateLimitMap.get(clientIP);
    const now = Date.now();
    const recentRequests = requests.filter(time => now - time < 10000); // Last 10 seconds
    
    return recentRequests.length > 20; // More than 20 requests in 10 seconds
  }

  async logReconAttempt(clientIP, userAgent, pathname, reconType) {
    try {
      await fetch(`${ORCHESTRATOR_ENDPOINT}/v1/camouflage/log`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${CAMOUFLAGE_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          timestamp: new Date().toISOString(),
          client_ip: clientIP,
          user_agent: userAgent,
          pathname: pathname,
          recon_type: reconType,
          cf_ray: request.headers.get('CF-Ray'),
          country: request.cf?.country
        })
      });
    } catch (error) {
      console.error('Failed to log reconnaissance attempt:', error);
    }
  }

  handleNormalRequest(request) {
    // Pass through to origin or return default response
    return fetch(request);
  }

  getFallbackTemplate(templateName) {
    // Fallback templates when orchestrator is unavailable
    const fallbackTemplates = {
      apache: {
        name: 'apache',
        version: '2.4.54',
        headers: {
          'Server': 'Apache/2.4.54 (Ubuntu)',
          'X-Powered-By': 'PHP/8.1.2'
        },
        error_pages: {
          '404': {
            body: '<html><head><title>404 Not Found</title></head><body><h1>Not Found</h1></body></html>',
            content_type: 'text/html'
          }
        },
        behavioral_patterns: {
          response_timing: { min_ms: 50, max_ms: 200 }
        }
      },
      nginx: {
        name: 'nginx',
        version: '1.22.1',
        headers: {
          'Server': 'nginx/1.22.1'
        },
        behavioral_patterns: {
          response_timing: { min_ms: 30, max_ms: 150 }
        }
      }
    };

    return fallbackTemplates[templateName] || fallbackTemplates.apache;
  }
}

// Cloudflare Worker event listener
addEventListener('fetch', event => {
  const worker = new CamouflageWorker();
  event.respondWith(worker.handleRequest(event.request));
});