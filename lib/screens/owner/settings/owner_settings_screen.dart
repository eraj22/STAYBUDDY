import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../api.dart';
import '../../owner_login_screen.dart';
import '../hostel/hostel_profile_editor_screen.dart';
import '../../../widgets/owner_feature_frame.dart';
import '../../../widgets/responsive_shell.dart';
import '../../../services/app_language.dart';

class OwnerSettingsScreen extends StatefulWidget {
  final String hostelName;
  final VoidCallback? onBackToDashboard;

  const OwnerSettingsScreen({
    super.key,
    required this.hostelName,
    this.onBackToDashboard,
  });

  @override
  State<OwnerSettingsScreen> createState() => _OwnerSettingsScreenState();
}

class _OwnerSettingsScreenState extends State<OwnerSettingsScreen> {
  int? _hostelId;
  bool _pushNotifications = true;
  bool _emailNotifications = true;
  bool _smsNotifications = false;

  bool _bookingAlerts = true;
  bool _complaintAlerts = true;
  bool _feeAlerts = true;
  bool _wardenActivityAlerts = false;

  bool _autoGenerateCredentials = true;
  bool _parentAccess = true;

  bool _maintenanceMode = false;

  String _selectedLanguage = 'English';
  String _selectedCurrency = 'PKR';
  String _selectedTheme = 'System';

  final List<String> _languages = ['English', 'Urdu'];

  final List<String> _currencies = ['PKR', 'USD'];

  final List<String> _themes = ['System', 'Light', 'Dark'];

  @override
  void initState() {
    super.initState();
    _selectedLanguage = AppLanguage.isUrdu ? 'Urdu' : 'English';
    _resolveHostel();
  }

  Future<void> _resolveHostel() async {
    try {
      final hostels = await Api().getOwnerHostels();
      final match = hostels.whereType<Map>().cast<Map>().firstWhere(
        (item) => item['name']?.toString().trim().toLowerCase() == widget.hostelName.trim().toLowerCase(),
        orElse: () => hostels.whereType<Map>().cast<Map>().isNotEmpty ? hostels.whereType<Map>().cast<Map>().first : <dynamic, dynamic>{},
      );
      final id = int.tryParse('${match['id'] ?? match['hostel_id']}');
      if (mounted && id != null) setState(() => _hostelId = id);
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: OwnerFeatureFrame(
          child: CustomScrollView(
            physics: const BouncingScrollPhysics(),
            slivers: [
              SliverToBoxAdapter(child: _buildHeader()),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 20, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'General',
                    subtitle: 'Basic hostel preferences',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildGeneralSettings()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Notifications',
                    subtitle: 'Choose how you receive updates',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildNotificationSettings()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Operational Alerts',
                    subtitle: 'Events that should notify the owner',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildOperationalAlerts()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Access & Security',
                    subtitle: 'Account and staff permissions',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildSecuritySettings()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Hostel Rules',
                    subtitle: 'Manage hostel-wide policies',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildRulesCard()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Data & Backup',
                    subtitle: 'Export and protect hostel data',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 18),
                sliver: SliverToBoxAdapter(child: _buildDataCard()),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(
                  child: _sectionTitle(
                    title: 'Advanced',
                    subtitle: 'System-level owner controls',
                  ),
                ),
              ),

              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 0, 18, 40),
                sliver: SliverToBoxAdapter(child: _buildAdvancedSettings()),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF073F42), Color(0xFF0A696D), Color(0xFF0A797B)],
        ),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: _goBackToDashboard,
                  ),
                  const Spacer(),
                  _circleButton(
                    icon: Icons.help_outline_rounded,
                    onTap: () {
                      _showTemporary(
                        'Help & support screen will be connected later.',
                      );
                    },
                  ),
                ],
              ),
              const SizedBox(height: 22),
              Text(
                'Hostel Settings',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 29,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 5),
              Text(
                widget.hostelName,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 12.5,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _circleButton({required IconData icon, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.18)),
          ),
          child: Icon(icon, color: Colors.white, size: 20),
        ),
      ),
    );
  }

  void _goBackToDashboard() {
    if (widget.onBackToDashboard != null) {
      widget.onBackToDashboard!();
      return;
    }
    Navigator.of(context).maybePop();
  }

  Widget _sectionTitle({required String title, required String subtitle}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.playfairDisplay(
            fontSize: 19,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(fontSize: 10.5, color: Colors.white70),
        ),
      ],
    );
  }

  Widget _buildGeneralSettings() {
    return _settingsCard(
      children: [
        _navigationTile(
          icon: Icons.apartment_rounded,
          title: 'Hostel Profile',
          subtitle: 'Name, address, type and contact details',
          color: AppTheme.teal,
          onTap: () {
            Navigator.of(context).push(MaterialPageRoute<void>(
              builder: (_) => HostelProfileEditorScreen(hostelName: widget.hostelName),
            ));
          },
        ),
        _divider(),
        _dropdownTile(
          icon: Icons.language_rounded,
          title: 'Language',
          value: _selectedLanguage,
          items: _languages,
          onChanged: (value) {
            if (value == null) return;

            setState(() {
              _selectedLanguage = value;
            });
            AppLanguage.set(value == 'Urdu' ? 'ur' : 'en');
          },
        ),
        _divider(),
        _dropdownTile(
          icon: Icons.currency_exchange_rounded,
          title: 'Currency',
          value: _selectedCurrency,
          items: _currencies,
          onChanged: (value) {
            if (value == null) return;

            setState(() {
              _selectedCurrency = value;
            });
          },
        ),
        _divider(),
        _dropdownTile(
          icon: Icons.palette_outlined,
          title: 'Appearance',
          value: _selectedTheme,
          items: _themes,
          onChanged: (value) {
            if (value == null) return;

            setState(() {
              _selectedTheme = value;
            });

            _showTemporary(
              'Theme preference saved in UI. Global theme connection comes later.',
            );
          },
        ),
      ],
    );
  }

  Widget _buildNotificationSettings() {
    return _settingsCard(
      children: [
        _switchTile(
          icon: Icons.notifications_active_outlined,
          title: 'Push Notifications',
          subtitle: 'Receive alerts inside the StayBuddy app',
          value: _pushNotifications,
          color: AppTheme.teal,
          onChanged: (value) {
            setState(() {
              _pushNotifications = value;
            });
          },
        ),
        _divider(),
        _switchTile(
          icon: Icons.email_outlined,
          title: 'Email Notifications',
          subtitle: 'Receive important alerts by email',
          value: _emailNotifications,
          color: const Color(0xFF3478B7),
          onChanged: (value) {
            setState(() {
              _emailNotifications = value;
            });
          },
        ),
        _divider(),
        _switchTile(
          icon: Icons.sms_outlined,
          title: 'SMS Notifications',
          subtitle: 'Receive critical alerts by SMS',
          value: _smsNotifications,
          color: const Color(0xFF7D55B4),
          onChanged: (value) {
            setState(() {
              _smsNotifications = value;
            });
          },
        ),
      ],
    );
  }

  Widget _buildOperationalAlerts() {
    return _settingsCard(
      children: [
        _switchTile(
          icon: Icons.person_add_alt_1_rounded,
          title: 'Booking Requests',
          subtitle: 'Notify when a new booking request arrives',
          value: _bookingAlerts,
          color: const Color(0xFF3D7B53),
          onChanged: (value) {
            setState(() {
              _bookingAlerts = value;
            });
          },
        ),
        _divider(),
        _switchTile(
          icon: Icons.report_problem_outlined,
          title: 'Complaint Alerts',
          subtitle: 'Notify when a new complaint is submitted',
          value: _complaintAlerts,
          color: const Color(0xFFC07421),
          onChanged: (value) {
            setState(() {
              _complaintAlerts = value;
            });
          },
        ),
        _divider(),
        _switchTile(
          icon: Icons.receipt_long_outlined,
          title: 'Fee Alerts',
          subtitle: 'Notify about due or late hostel payments',
          value: _feeAlerts,
          color: const Color(0xFF3478B7),
          onChanged: (value) {
            setState(() {
              _feeAlerts = value;
            });
          },
        ),
        _divider(),
        _switchTile(
          icon: Icons.badge_outlined,
          title: 'Warden Activity',
          subtitle: 'Receive alerts for important staff actions',
          value: _wardenActivityAlerts,
          color: AppTheme.teal,
          onChanged: (value) {
            setState(() {
              _wardenActivityAlerts = value;
            });
          },
        ),
      ],
    );
  }

  Widget _buildSecuritySettings() {
    return _settingsCard(
      children: [
        _switchTile(
          icon: Icons.password_rounded,
          title: 'Auto Generate Credentials',
          subtitle: 'Create logins when students or wardens are added',
          value: _autoGenerateCredentials,
          color: AppTheme.teal,
          onChanged: (value) {
            setState(() {
              _autoGenerateCredentials = value;
            });
          },
        ),
        _divider(),
        _switchTile(
          icon: Icons.family_restroom_rounded,
          title: 'Parent Portal Access',
          subtitle: 'Allow linked parents to access student information',
          value: _parentAccess,
          color: const Color(0xFF3478B7),
          onChanged: (value) {
            setState(() {
              _parentAccess = value;
            });
          },
        ),
        _divider(),
        _navigationTile(
          icon: Icons.key_rounded,
          title: 'Change Owner Password',
          subtitle: 'Update the owner account password',
          color: const Color(0xFFC07421),
          onTap: () {
            _showChangePasswordDialog();
          },
        ),
        _divider(),
        _navigationTile(
          icon: Icons.devices_rounded,
          title: 'Logged In Devices',
          subtitle: 'Review active owner sessions',
          color: const Color(0xFF52616B),
          onTap: () {
            _showDevicesSheet();
          },
        ),
      ],
    );
  }

  Widget _buildRulesCard() {
    return _settingsCard(
      children: [
        _navigationTile(
          icon: Icons.rule_folder_outlined,
          title: 'Hostel Rules',
          subtitle: 'Curfew, visitors, conduct and other policies',
          color: AppTheme.teal,
          onTap: () {
            _showRulesSheet();
          },
        ),
        _divider(),
        _navigationTile(
          icon: Icons.restaurant_outlined,
          title: 'Mess Timings',
          subtitle: 'Configure breakfast, lunch and dinner timings',
          color: const Color(0xFF3D7B53),
          onTap: () {
            _showPolicyEditor('Mess timings', 'mess_timings');
          },
        ),
        _divider(),
        _navigationTile(
          icon: Icons.schedule_rounded,
          title: 'Entry / Exit Timings',
          subtitle: 'Set hostel gate and curfew timings',
          color: const Color(0xFF3478B7),
          onTap: () {
            _showPolicyEditor('Entry / exit timings', 'gate_policy');
          },
        ),
        _divider(),
        _navigationTile(
          icon: Icons.notifications_none_rounded,
          title: 'Announcement Policy',
          subtitle: 'Configure who can publish hostel announcements',
          color: const Color(0xFF7D55B4),
          onTap: () {
            _showPolicyEditor('Announcement policy', 'announcement_policy');
          },
        ),
      ],
    );
  }

  Widget _buildDataCard() {
    return _settingsCard(
      children: [
        _navigationTile(
          icon: Icons.cloud_sync_outlined,
          title: 'Backup Hostel Data',
          subtitle: 'Create a backup of hostel records',
          color: AppTheme.teal,
          onTap: () {
            _showTemporary(
              'Backup functionality will be connected with backend later.',
            );
          },
        ),
        _divider(),
        _navigationTile(
          icon: Icons.download_outlined,
          title: 'Export All Data',
          subtitle: 'Download students, rooms and reports',
          color: const Color(0xFF3478B7),
          onTap: () {
            _showExportSheet();
          },
        ),
        _divider(),
        _navigationTile(
          icon: Icons.history_rounded,
          title: 'Audit Log',
          subtitle: 'Review important owner and staff actions',
          color: const Color(0xFF7D55B4),
          onTap: () {
            _showAuditLog();
          },
        ),
      ],
    );
  }

  Widget _buildAdvancedSettings() {
    return Column(
      children: [
        _settingsCard(
          children: [
            _switchTile(
              icon: Icons.construction_rounded,
              title: 'Maintenance Mode',
              subtitle: 'Temporarily limit hostel operations',
              value: _maintenanceMode,
              color: const Color(0xFFC07421),
              onChanged: (value) {
                if (value) {
                  _confirmMaintenanceMode();
                } else {
                  setState(() {
                    _maintenanceMode = false;
                  });
                }
              },
            ),
          ],
        ),

        const SizedBox(height: 16),

        Container(
          padding: const EdgeInsets.all(17),
          decoration: BoxDecoration(
            color: Colors.red.shade50,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.red.shade100),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(
                    Icons.warning_amber_rounded,
                    color: Colors.red.shade600,
                    size: 20,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    'Danger Zone',
                    style: GoogleFonts.dmSans(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: Colors.red.shade700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 7),
              Text(
                'These actions can affect access to your hostel account and data.',
                style: GoogleFonts.dmSans(
                  fontSize: 10.5,
                  height: 1.5,
                  color: Colors.red.shade500,
                ),
              ),
              const SizedBox(height: 14),
              SizedBox(
                width: double.infinity,
                height: 46,
                child: OutlinedButton.icon(
                  onPressed: _showLogoutDialog,
                  icon: const Icon(Icons.logout_rounded, size: 17),
                  label: Text(
                    'Logout Owner Account',
                    style: GoogleFonts.dmSans(
                      fontSize: 11.5,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.red.shade600,
                    side: BorderSide(color: Colors.red.shade300),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _settingsCard({required List<Widget> children}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
      ),
      child: Column(children: children),
    );
  }

  Widget _navigationTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(14),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Row(
            children: [
              Container(
                width: 41,
                height: 41,
                decoration: BoxDecoration(
                  color: color.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: color, size: 19),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: GoogleFonts.dmSans(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w600,
                        color: OwnerFeaturePalette.text,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      style: GoogleFonts.dmSans(
                        fontSize: 9.5,
                        height: 1.35,
                        color: OwnerFeaturePalette.textMuted,
                      ),
                    ),
                  ],
                ),
              ),
              Icon(
                Icons.chevron_right_rounded,
                size: 20,
                color: OwnerFeaturePalette.textMuted,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _switchTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required Color color,
    required ValueChanged<bool> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 11),
      child: Row(
        children: [
          Container(
            width: 41,
            height: 41,
            decoration: BoxDecoration(
              color: color.withOpacity(0.08),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: color, size: 19),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: OwnerFeaturePalette.text,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    height: 1.35,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ),
              ],
            ),
          ),
          Switch.adaptive(
            value: value,
            activeColor: AppTheme.teal,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  Widget _dropdownTile({
    required IconData icon,
    required String title,
    required String value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Container(
            width: 41,
            height: 41,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.07),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: AppTheme.teal, size: 19),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              title,
              style: GoogleFonts.dmSans(
                fontSize: 12.5,
                fontWeight: FontWeight.w600,
                color: OwnerFeaturePalette.text,
              ),
            ),
          ),
          DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: value,
              borderRadius: BorderRadius.circular(16),
              style: GoogleFonts.dmSans(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: AppTheme.teal,
              ),
              items: items.map((item) {
                return DropdownMenuItem(value: item, child: Text(item));
              }).toList(),
              onChanged: onChanged,
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      thickness: 0.6,
      color: OwnerFeaturePalette.border,
    );
  }

  void _showChangePasswordDialog() {
    final currentCtrl = TextEditingController();
    final newCtrl = TextEditingController();
    final confirmCtrl = TextEditingController();

    showDialog(
      context: context,
      builder: (dialogContext) {
        return Dialog(
          backgroundColor: OwnerFeaturePalette.surfaceStrong,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(26),
          ),
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Change Password',
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: OwnerFeaturePalette.text,
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    'Use a strong password for your owner account.',
                    style: GoogleFonts.dmSans(
                      fontSize: 10.5,
                      color: OwnerFeaturePalette.textMuted,
                    ),
                  ),
                  const SizedBox(height: 18),
                  _dialogField(
                    controller: currentCtrl,
                    hint: 'Current password',
                    icon: Icons.lock_outline_rounded,
                    obscure: true,
                  ),
                  const SizedBox(height: 10),
                  _dialogField(
                    controller: newCtrl,
                    hint: 'New password',
                    icon: Icons.password_rounded,
                    obscure: true,
                  ),
                  const SizedBox(height: 10),
                  _dialogField(
                    controller: confirmCtrl,
                    hint: 'Confirm new password',
                    icon: Icons.verified_user_outlined,
                    obscure: true,
                  ),
                  const SizedBox(height: 18),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () {
                            Navigator.pop(dialogContext);
                          },
                          style: OutlinedButton.styleFrom(
                            foregroundColor: OwnerFeaturePalette.textMuted,
                            minimumSize: const Size.fromHeight(48),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(15),
                            ),
                          ),
                          child: const Text('Cancel'),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: ElevatedButton(
                          onPressed: () {
                            final current = currentCtrl.text.trim();
                            final newPassword = newCtrl.text.trim();
                            final confirm = confirmCtrl.text.trim();

                            if (current.isEmpty ||
                                newPassword.isEmpty ||
                                confirm.isEmpty) {
                              _showTemporary(
                                'Please fill all password fields.',
                              );
                              return;
                            }

                            if (newPassword != confirm) {
                              _showTemporary('New passwords do not match.');
                              return;
                            }

                            Navigator.pop(dialogContext);

                            _showTemporary(
                              'Password update will be connected to backend later.',
                            );
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppTheme.teal,
                            foregroundColor: Colors.white,
                            elevation: 0,
                            minimumSize: const Size.fromHeight(48),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(15),
                            ),
                          ),
                          child: const Text('Update'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _dialogField({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
    bool obscure = false,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: TextField(
        controller: controller,
        obscureText: obscure,
        style: GoogleFonts.dmSans(
          fontSize: 13,
          color: OwnerFeaturePalette.text,
        ),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
            fontSize: 12,
            color: OwnerFeaturePalette.textMuted,
          ),
          prefixIcon: Icon(
            icon,
            size: 19,
            color: OwnerFeaturePalette.textMuted,
          ),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(vertical: 15),
        ),
      ),
    );
  }

  Future<void> _showDevicesSheet() async {
    List<dynamic> sessions;
    try {
      sessions = await Api().getOwnerSessions();
    } catch (error) {
      _showTemporary(error.toString());
      return;
    }
    if (!mounted) return;
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(18, 12, 18, 20),
            decoration: BoxDecoration(
              color: OwnerFeaturePalette.surfaceStrong,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 42,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 18),
                    decoration: BoxDecoration(
                      color: OwnerFeaturePalette.border,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ),
                Text(
                  'Logged In Devices',
                  style: GoogleFonts.playfairDisplay(
                    fontSize: 23,
                    fontWeight: FontWeight.w700,
                    color: OwnerFeaturePalette.text,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  'Owner account sessions',
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ),
                const SizedBox(height: 18),
                if (sessions.isEmpty)
                  const Padding(padding: EdgeInsets.all(18), child: Text('No active sessions found.'))
                else
                  ...sessions.map((raw) {
                    final session = Map<String, dynamic>.from(raw as Map);
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: _deviceTile(
                        icon: Icons.devices_rounded,
                        title: session['user_agent']?.toString().isNotEmpty == true ? session['user_agent'].toString() : 'StayBuddy device',
                        subtitle: '${session['ip_address'] ?? 'Unknown network'} · Last active ${session['last_seen_at'] ?? session['created_at'] ?? ''}',
                        current: session['current'] == true,
                        onRevoke: () async {
                          try { await Api().revokeOwnerSession(session['id'].toString()); if (context.mounted) Navigator.pop(context); _showTemporary('Device signed out.'); } catch (error) { _showTemporary(error.toString()); }
                        },
                      ),
                    );
                  }),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _deviceTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool current,
    VoidCallback? onRevoke,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.07),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: AppTheme.teal, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: OwnerFeaturePalette.text,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ),
              ],
            ),
          ),
          if (current)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: const Color(0xFFEDF6EF),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Current',
                style: GoogleFonts.dmSans(
                  fontSize: 8.5,
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF3D7B53),
                ),
              ),
            )
          else
            IconButton(
              onPressed: onRevoke,
              icon: Icon(
                Icons.logout_rounded,
                size: 18,
                color: Colors.red.shade500,
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _showRulesSheet() async {
    if (_hostelId == null) await _resolveHostel();
    if (_hostelId == null) { _showTemporary('Your hostel is still loading. Please try again.'); return; }
    List<String> rules = [];
    try {
      final settings = await Api().getOwnerHostelRules(_hostelId!);
      rules = List<dynamic>.from(settings['rules'] ?? const []).map((item) => item.toString()).toList();
    } catch (error) { _showTemporary(error.toString()); return; }
    final controller = TextEditingController(text: rules.join('\n'));

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(18, 12, 18, 20),
            decoration: BoxDecoration(
              color: OwnerFeaturePalette.surfaceStrong,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 42,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 18),
                    decoration: BoxDecoration(
                      color: OwnerFeaturePalette.border,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ),
                Text(
                  'Hostel Rules',
                  style: GoogleFonts.playfairDisplay(
                    fontSize: 23,
                    fontWeight: FontWeight.w700,
                    color: OwnerFeaturePalette.text,
                  ),
                ),
                const SizedBox(height: 5),
                Text(
                  'Write one rule on each line',
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ),
                const SizedBox(height: 18),
                TextField(controller: controller, minLines: 6, maxLines: 12, decoration: const InputDecoration(border: OutlineInputBorder(), hintText: 'One rule per line')),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  height: 48,
                  child: ElevatedButton.icon(
                    onPressed: () async {
                      final updated = controller.text.split('\n').map((item) => item.trim()).where((item) => item.isNotEmpty).toList();
                      try { await Api().saveOwnerHostelRules(_hostelId!, rules: updated); if (context.mounted) Navigator.pop(context); _showTemporary('Hostel rules saved.'); } catch (error) { _showTemporary(error.toString()); }
                    },
                    icon: const Icon(Icons.edit_note_rounded, size: 19),
                    label: Text(
                      'Manage Rules',
                      style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.teal,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(15),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _showPolicyEditor(String title, String settingKey) async {
    if (_hostelId == null) await _resolveHostel();
    if (_hostelId == null) {
      _showTemporary('Your hostel is still loading. Please try again.');
      return;
    }
    try {
      final settings = await Api().getOwnerHostelRules(_hostelId!);
      final current = settings[settingKey] is Map
          ? Map<String, dynamic>.from(settings[settingKey] as Map)
          : <String, dynamic>{};
      final controller = TextEditingController(text: current['notes']?.toString() ?? '');
      if (!mounted) return;
      await showDialog<void>(
        context: context,
        builder: (context) => AlertDialog(
          title: Text(title),
          content: TextField(
            controller: controller,
            minLines: 3,
            maxLines: 6,
            decoration: const InputDecoration(
              labelText: 'Policy details',
              hintText: 'For example: Gate opens at 7:00 AM and closes at 9:00 PM.',
            ),
          ),
          actions: [
            TextButton(onPressed: () => Navigator.pop(context), child: const Text('Cancel')),
            FilledButton(
              onPressed: () async {
                final updated = Map<String, dynamic>.from(current)..['notes'] = controller.text.trim();
                final rules = List<dynamic>.from(settings['rules'] ?? const []).map((item) => item.toString()).toList();
                final mess = settingKey == 'mess_timings' ? updated : Map<String, dynamic>.from(settings['mess_timings'] as Map? ?? const {});
                final gate = settingKey == 'gate_policy' ? updated : Map<String, dynamic>.from(settings['gate_policy'] as Map? ?? const {});
                final announcements = settingKey == 'announcement_policy' ? updated : Map<String, dynamic>.from(settings['announcement_policy'] as Map? ?? const {});
                await Api().saveOwnerHostelRules(_hostelId!, rules: rules, messTimings: mess, gatePolicy: gate, announcementPolicy: announcements);
                if (context.mounted) Navigator.pop(context);
                _showTemporary('$title saved.');
              },
              child: const Text('Save'),
            ),
          ],
        ),
      );
    } catch (error) {
      _showTemporary(error.toString());
    }
  }

  void _showExportSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: OwnerFeaturePalette.surfaceStrong,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _bottomMenuTile(
                  icon: Icons.groups_outlined,
                  title: 'Student Records',
                  subtitle: 'Export student data',
                  onTap: () {
                    Navigator.pop(context);
                    _export('students');
                  },
                ),
                _bottomMenuTile(
                  icon: Icons.bed_outlined,
                  title: 'Room Records',
                  subtitle: 'Export rooms and bed assignments',
                  onTap: () {
                    Navigator.pop(context);
                    _export('rooms');
                  },
                ),
                _bottomMenuTile(
                  icon: Icons.insights_outlined,
                  title: 'Analytics Report',
                  subtitle: 'Export owner analytics',
                  onTap: () {
                    Navigator.pop(context);
                    _export('analytics');
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _bottomMenuTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      leading: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.07),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, color: AppTheme.teal, size: 20),
      ),
      title: Text(
        title,
        style: GoogleFonts.dmSans(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
          color: OwnerFeaturePalette.text,
        ),
      ),
      subtitle: Text(
        subtitle,
        style: GoogleFonts.dmSans(
          fontSize: 9.5,
          color: OwnerFeaturePalette.textMuted,
        ),
      ),
      trailing: const Icon(Icons.chevron_right_rounded),
    );
  }

  void _confirmMaintenanceMode() {
    showDialog(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: OwnerFeaturePalette.surfaceStrong,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(22),
          ),
          title: Text(
            'Enable Maintenance Mode?',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
          content: Text(
            'Some hostel operations may become temporarily unavailable while maintenance mode is active.',
            style: GoogleFonts.dmSans(
              fontSize: 11.5,
              height: 1.5,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);
              },
              child: Text(
                'Cancel',
                style: GoogleFonts.dmSans(color: OwnerFeaturePalette.textMuted),
              ),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.pop(dialogContext);

                setState(() {
                  _maintenanceMode = true;
                });
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC07421),
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(13),
                ),
              ),
              child: Text(
                'Enable',
                style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
              ),
            ),
          ],
        );
      },
    );
  }

  void _showLogoutDialog() {
    showDialog(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: OwnerFeaturePalette.surfaceStrong,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(22),
          ),
          title: Text(
            'Logout?',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
          content: Text(
            'You will need to enter your owner credentials again to access StayBuddy.',
            style: GoogleFonts.dmSans(
              fontSize: 11.5,
              height: 1.5,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(dialogContext);
              },
              child: Text(
                'Cancel',
                style: GoogleFonts.dmSans(color: OwnerFeaturePalette.textMuted),
              ),
            ),
            ElevatedButton(
              onPressed: () async {
                Navigator.pop(dialogContext);
                await Api().logout();
                if (!mounted) return;
                Navigator.of(context).pushAndRemoveUntil(
                  MaterialPageRoute<void>(builder: (_) => const OwnerLoginScreen()),
                  (_) => false,
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red.shade600,
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(13),
                ),
              ),
              child: Text(
                'Logout',
                style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
              ),
            ),
          ],
        );
      },
    );
  }

  Future<void> _export(String kind) async {
    try {
      final data = await Api().getOwnerExportCsv(kind);
      await Clipboard.setData(ClipboardData(text: data));
      if (mounted) Navigator.of(context).maybePop();
      _showTemporary('${kind[0].toUpperCase()}${kind.substring(1)} CSV copied to the clipboard.');
    } catch (error) {
      _showTemporary(error.toString());
    }
  }

  Future<void> _revokeOtherSessions() async {
    try {
      await Api().revokeOtherOwnerSessions();
      _showTemporary('All other owner devices have been signed out.');
    } catch (error) {
      _showTemporary(error.toString());
    }
  }

  Future<void> _showAuditLog() async {
    try {
      final events = await Api().getOwnerAuditEvents();
      if (!mounted) return;
      showModalBottomSheet(
        context: context,
        isScrollControlled: true,
        backgroundColor: Colors.transparent,
        builder: (_) => SafeArea(
          top: false,
          child: Container(
            height: MediaQuery.of(context).size.height * .7,
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(color: OwnerFeaturePalette.surfaceStrong, borderRadius: BorderRadius.circular(26)),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('Audit Log', style: GoogleFonts.playfairDisplay(fontSize: 23, fontWeight: FontWeight.w700, color: OwnerFeaturePalette.text)),
              const SizedBox(height: 6),
              Text('The latest 200 protected owner actions.', style: GoogleFonts.dmSans(fontSize: 11, color: OwnerFeaturePalette.textMuted)),
              const SizedBox(height: 14),
              Expanded(child: events.isEmpty ? const Center(child: Text('No owner actions recorded yet.')) : ListView.separated(
                itemCount: events.length,
                separatorBuilder: (_, __) => const Divider(height: 1),
                itemBuilder: (_, index) {
                  final event = Map<String, dynamic>.from(events[index] as Map);
                  final who = event['target_name']?.toString();
                  return ListTile(
                    dense: true,
                    leading: const Icon(Icons.history_rounded, color: AppTheme.teal),
                    title: Text(event['event_type']?.toString().replaceAll('_', ' ') ?? 'Owner action', style: GoogleFonts.dmSans(fontWeight: FontWeight.w700)),
                    subtitle: Text('${who?.isNotEmpty == true ? '$who · ' : ''}${event['created_at'] ?? ''}', maxLines: 2, overflow: TextOverflow.ellipsis),
                  );
                },
              )),
            ]),
          ),
        ),
      );
    } catch (error) { _showTemporary(error.toString()); }
  }

  void _showTemporary(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Text(
            message,
            style: GoogleFonts.dmSans(color: Colors.white),
          ),
        ),
      );
  }
}
